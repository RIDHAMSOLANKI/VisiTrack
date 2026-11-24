const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const mongoose = require('mongoose');
const { Admin, Visitor, Visit } = require('./models');
const { signToken, requireAdmin, requireGateOrAdmin, bcrypt } = require('./auth');
const { MLError } = require('./mlClient');

const MAX_IMAGES = 10;

/**
 * Build the Express app.
 * ml:      ML client (see mlClient.js), injected so tests can fake it
 * emit:    (event, payload) => void, pushes live updates to dashboards (Socket.IO)
 */
function createApp({ ml, emit = () => {}, jwtSecret, gateKey, cooldownSec = 60, corsOrigin = '*', log = true }) {
  if (!jwtSecret) throw new Error('JWT_SECRET is required');
  const app = express();
  app.use(cors({ origin: corsOrigin }));
  app.use(express.json({ limit: '25mb' })); // webcam frames arrive as base64
  if (log) app.use(morgan('dev'));

  const admin = requireAdmin(jwtSecret);
  const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
  const isId = (id) => mongoose.isValidObjectId(id);

  // ------------------------------------------------------------------ health
  app.get('/api/health', wrap(async (req, res) => {
    let mlStatus;
    try {
      mlStatus = await ml.health();
    } catch (err) {
      mlStatus = { status: 'down', error: err.message };
    }
    res.json({ status: 'ok', db: mongoose.connection.readyState === 1 ? 'up' : 'down', ml: mlStatus });
  }));

  // ------------------------------------------------------------------ auth
  app.post('/api/auth/login', wrap(async (req, res) => {
    const { email, password } = req.body || {};
    const found = email && (await Admin.findOne({ email: String(email).toLowerCase() }));
    if (!found || !(await bcrypt.compare(String(password || ''), found.passwordHash))) {
      return res.status(401).json({ error: 'Wrong email or password' });
    }
    res.json({ token: signToken(found, jwtSecret), admin: { email: found.email, name: found.name } });
  }));

  // ------------------------------------------------------------------ visitors
  app.get('/api/visitors', admin, wrap(async (req, res) => {
    const q = String(req.query.q || '').trim();
    const filter = { active: true };
    if (q) {
      const rx = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
      filter.$or = [{ name: rx }, { phone: rx }, { email: rx }, { host: rx }];
    }
    const visitors = await Visitor.find(filter).sort({ createdAt: -1 }).limit(200).lean();
    res.json({ visitors });
  }));

  app.post('/api/visitors', admin, wrap(async (req, res) => {
    const { name, phone, email, purpose, host, images } = req.body || {};
    if (!name || !String(name).trim()) return res.status(400).json({ error: 'Name is required' });
    if (!Array.isArray(images) || images.length === 0) {
      return res.status(400).json({ error: 'At least one face photo is required' });
    }
    if (images.length > MAX_IMAGES) return res.status(400).json({ error: `At most ${MAX_IMAGES} photos` });

    const visitor = await Visitor.create({ name, phone, email, purpose, host });
    try {
      const result = await ml.register(String(visitor._id), images);
      visitor.faceCount = result.stored;
      await visitor.save();
      res.status(201).json({ visitor, facesUsed: result.faces_used, skipped: result.skipped });
    } catch (err) {
      await Visitor.deleteOne({ _id: visitor._id }); // no half-registered visitors
      throw err;
    }
  }));

  app.get('/api/visitors/:id', admin, wrap(async (req, res) => {
    if (!isId(req.params.id)) return res.status(404).json({ error: 'Visitor not found' });
    const visitor = await Visitor.findById(req.params.id).lean();
    if (!visitor) return res.status(404).json({ error: 'Visitor not found' });
    const visits = await Visit.find({ visitor: visitor._id }).sort({ checkInAt: -1 }).limit(50).lean();
    res.json({ visitor, visits });
  }));

  app.post('/api/visitors/:id/faces', admin, wrap(async (req, res) => {
    if (!isId(req.params.id)) return res.status(404).json({ error: 'Visitor not found' });
    const visitor = await Visitor.findById(req.params.id);
    if (!visitor || !visitor.active) return res.status(404).json({ error: 'Visitor not found' });
    const { images, replace } = req.body || {};
    if (!Array.isArray(images) || !images.length || images.length > MAX_IMAGES) {
      return res.status(400).json({ error: `Send 1-${MAX_IMAGES} photos` });
    }
    const result = await ml.register(String(visitor._id), images, Boolean(replace));
    visitor.faceCount = result.stored;
    await visitor.save();
    res.json({ visitor, facesUsed: result.faces_used, skipped: result.skipped });
  }));

  app.delete('/api/visitors/:id', admin, wrap(async (req, res) => {
    if (!isId(req.params.id)) return res.status(404).json({ error: 'Visitor not found' });
    const visitor = await Visitor.findById(req.params.id);
    if (!visitor) return res.status(404).json({ error: 'Visitor not found' });
    await ml.remove(String(visitor._id)); // delete face data first (privacy)
    await Visitor.deleteOne({ _id: visitor._id }); // visit history keeps the name copy
    res.json({ deleted: true });
  }));

  // ------------------------------------------------------------------ gate check-in
  app.post('/api/checkin', requireGateOrAdmin(jwtSecret, gateKey), wrap(async (req, res) => {
    const { image } = req.body || {};
    const gate = String(req.body?.gate || 'main').slice(0, 50);
    if (!image) return res.status(400).json({ error: 'image is required' });

    const result = await ml.identify(image);
    if (!result.face_found) return res.json({ status: 'no_face' });

    let visitor = null;
    if (result.recognized && isId(result.visitor_id)) {
      visitor = await Visitor.findOne({ _id: result.visitor_id, active: true }).lean();
    }
    const status = visitor ? 'recognized' : 'unknown';

    // The kiosk sends a frame every ~2 s, so ignore repeats within the cooldown.
    const since = new Date(Date.now() - cooldownSec * 1000);
    const recent = await Visit.findOne({
      gate, status, visitor: visitor ? visitor._id : null, checkInAt: { $gte: since },
    }).sort({ checkInAt: -1 }).lean();
    if (recent) return res.json({ status, duplicate: true, visit: recent, box: result.box });

    const visit = await Visit.create({
      visitor: visitor ? visitor._id : null,
      visitorName: visitor ? visitor.name : 'Unknown',
      status, gate, score: result.score, matchRatio: result.match_ratio,
    });
    emit('visit:new', visit.toObject());
    res.status(201).json({ status, visit, visitor, box: result.box });
  }));

  // ------------------------------------------------------------------ visits
  app.get('/api/visits', admin, wrap(async (req, res) => {
    const { status, gate, from, to, q, visitor } = req.query;
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit, 10) || 25));
    const filter = {};
    if (status === 'recognized' || status === 'unknown') filter.status = status;
    if (gate) filter.gate = String(gate);
    if (visitor && isId(visitor)) filter.visitor = visitor;
    if (q) filter.visitorName = new RegExp(String(q).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
    if (from || to) {
      filter.checkInAt = {};
      if (from && !isNaN(Date.parse(from))) filter.checkInAt.$gte = new Date(from);
      if (to && !isNaN(Date.parse(to))) filter.checkInAt.$lte = new Date(to);
    }
    const [visits, total] = await Promise.all([
      Visit.find(filter).sort({ checkInAt: -1 }).skip((page - 1) * limit).limit(limit).lean(),
      Visit.countDocuments(filter),
    ]);
    res.json({ visits, total, page, pages: Math.ceil(total / limit) });
  }));

  app.patch('/api/visits/:id/checkout', admin, wrap(async (req, res) => {
    if (!isId(req.params.id)) return res.status(404).json({ error: 'Visit not found' });
    const visit = await Visit.findOneAndUpdate(
      { _id: req.params.id, checkOutAt: null },
      { checkOutAt: new Date() },
      { new: true }
    ).lean();
    if (!visit) return res.status(404).json({ error: 'Visit not found or already checked out' });
    emit('visit:updated', visit);
    res.json({ visit });
  }));

  // ------------------------------------------------------------------ stats
  app.get('/api/stats', admin, wrap(async (req, res) => {
    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);
    const weekAgo = new Date(startOfDay.getTime() - 6 * 86400000);

    const [today, unknownToday, inside, visitors, perDay] = await Promise.all([
      Visit.countDocuments({ checkInAt: { $gte: startOfDay } }),
      Visit.countDocuments({ checkInAt: { $gte: startOfDay }, status: 'unknown' }),
      Visit.countDocuments({ checkInAt: { $gte: startOfDay }, status: 'recognized', checkOutAt: null }),
      Visitor.countDocuments({ active: true }),
      Visit.aggregate([
        { $match: { checkInAt: { $gte: weekAgo } } },
        { $group: { _id: { $dateToString: { format: '%Y-%m-%d', date: '$checkInAt' } }, count: { $sum: 1 } } },
        { $sort: { _id: 1 } },
      ]),
    ]);
    res.json({ today, unknownToday, inside, visitors, perDay });
  }));

  // ------------------------------------------------------------------ errors
  app.use('/api', (req, res) => res.status(404).json({ error: 'Not found' }));
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err instanceof MLError) return res.status(err.status).json({ error: err.message });
    if (err.name === 'ValidationError') return res.status(400).json({ error: err.message });
    if (err.type === 'entity.too.large') return res.status(413).json({ error: 'Request too large' });
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  });

  return app;
}

module.exports = { createApp };
