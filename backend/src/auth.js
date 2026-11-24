const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { Admin } = require('./models');

function signToken(admin, secret) {
  return jwt.sign({ sub: String(admin._id), email: admin.email }, secret, { expiresIn: '12h' });
}

function requireAdmin(secret) {
  return (req, res, next) => {
    const header = req.headers.authorization || '';
    const token = header.startsWith('Bearer ') ? header.slice(7) : null;
    if (!token) return res.status(401).json({ error: 'Login required' });
    try {
      req.admin = jwt.verify(token, secret);
      return next();
    } catch {
      return res.status(401).json({ error: 'Session expired, please log in again' });
    }
  };
}

function safeEqual(a, b) {
  const x = Buffer.from(String(a));
  const y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

// Gate kiosks use a shared key instead of an admin login. Admins may also call check-in.
function requireGateOrAdmin(secret, gateKey) {
  const admin = requireAdmin(secret);
  return (req, res, next) => {
    const key = req.headers['x-gate-key'];
    if (gateKey && key && safeEqual(key, gateKey)) return next();
    return admin(req, res, next);
  };
}

async function seedAdmin(email, password) {
  if (!email || !password) return null;
  if (await Admin.exists({})) return null;
  const passwordHash = await bcrypt.hash(password, 10);
  return Admin.create({ email, passwordHash });
}

module.exports = { signToken, requireAdmin, requireGateOrAdmin, seedAdmin, bcrypt };
