const request = require('supertest');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const { createApp } = require('../src/app');
const { seedAdmin } = require('../src/auth');
const { Visit } = require('../src/models');

let mongo;
let app;
let token;
let events;
const ml = {
  registered: {},
  nextIdentify: null,
  async health() { return { status: 'ok' }; },
  async register(id, images) {
    this.registered[id] = (this.registered[id] || 0) + images.length;
    return { visitor_id: id, faces_used: images.length, skipped: 0, stored: this.registered[id] };
  },
  async identify() { return this.nextIdentify; },
  async remove(id) { delete this.registered[id]; return { removed: true }; },
};

beforeAll(async () => {
  mongo = await MongoMemoryServer.create();
  await mongoose.connect(mongo.getUri());
  await seedAdmin('admin@test.local', 'pass1234');
});

afterAll(async () => {
  await mongoose.disconnect();
  await mongo.stop();
});

beforeEach(async () => {
  events = [];
  app = createApp({
    ml, jwtSecret: 'test-secret', gateKey: 'gate-key', cooldownSec: 60, log: false,
    emit: (e, p) => events.push([e, p]),
  });
  const res = await request(app).post('/api/auth/login').send({ email: 'admin@test.local', password: 'pass1234' });
  token = res.body.token;
});

const auth = () => ({ Authorization: `Bearer ${token}` });

test('login rejects a wrong password', async () => {
  const res = await request(app).post('/api/auth/login').send({ email: 'admin@test.local', password: 'nope' });
  expect(res.status).toBe(401);
});

test('admin routes need a token', async () => {
  expect((await request(app).get('/api/visitors')).status).toBe(401);
});

test('register visitor, recognise at gate, ignore repeat, check out', async () => {
  const created = await request(app).post('/api/visitors').set(auth())
    .send({ name: 'Asha Rao', host: 'Prof. X', images: ['img1', 'img2'] });
  expect(created.status).toBe(201);
  const id = created.body.visitor._id;
  expect(created.body.visitor.faceCount).toBe(2);

  ml.nextIdentify = { face_found: true, recognized: true, visitor_id: id, score: 0.93, match_ratio: 1, box: {} };
  const first = await request(app).post('/api/checkin').set('X-Gate-Key', 'gate-key').send({ image: 'x', gate: 'main' });
  expect(first.status).toBe(201);
  expect(first.body.status).toBe('recognized');
  expect(first.body.visit.visitorName).toBe('Asha Rao');
  expect(events[0][0]).toBe('visit:new');

  const again = await request(app).post('/api/checkin').set('X-Gate-Key', 'gate-key').send({ image: 'x', gate: 'main' });
  expect(again.body.duplicate).toBe(true);
  expect(await Visit.countDocuments({ visitor: id })).toBe(1);

  const out = await request(app).patch(`/api/visits/${first.body.visit._id}/checkout`).set(auth());
  expect(out.status).toBe(200);
  expect(out.body.visit.checkOutAt).toBeTruthy();

  const stats = await request(app).get('/api/stats').set(auth());
  expect(stats.body.today).toBeGreaterThanOrEqual(1);
});

test('unknown face is logged as unknown; no face is not logged', async () => {
  ml.nextIdentify = { face_found: true, recognized: false, visitor_id: null, score: 0.2, match_ratio: 0, box: {} };
  const res = await request(app).post('/api/checkin').set('X-Gate-Key', 'gate-key').send({ image: 'x', gate: 'side' });
  expect(res.body.status).toBe('unknown');

  ml.nextIdentify = { face_found: false, recognized: false };
  const none = await request(app).post('/api/checkin').set('X-Gate-Key', 'gate-key').send({ image: 'x', gate: 'side' });
  expect(none.body.status).toBe('no_face');

  const list = await request(app).get('/api/visits?status=unknown&gate=side').set(auth());
  expect(list.body.total).toBe(1);
});

test('wrong gate key is rejected', async () => {
  const res = await request(app).post('/api/checkin').set('X-Gate-Key', 'bad').send({ image: 'x' });
  expect(res.status).toBe(401);
});

test('deleting a visitor removes face data but keeps visit history', async () => {
  const created = await request(app).post('/api/visitors').set(auth()).send({ name: 'Temp', images: ['a'] });
  const id = created.body.visitor._id;
  ml.nextIdentify = { face_found: true, recognized: true, visitor_id: id, score: 0.9, match_ratio: 1, box: {} };
  await request(app).post('/api/checkin').set('X-Gate-Key', 'gate-key').send({ image: 'x', gate: 'g2' });

  const del = await request(app).delete(`/api/visitors/${id}`).set(auth());
  expect(del.body.deleted).toBe(true);
  expect(ml.registered[id]).toBeUndefined();
  expect(await Visit.countDocuments({ visitorName: 'Temp' })).toBe(1);
});

test('validation errors', async () => {
  expect((await request(app).post('/api/visitors').set(auth()).send({ images: ['a'] })).status).toBe(400);
  expect((await request(app).post('/api/visitors').set(auth()).send({ name: 'A', images: [] })).status).toBe(400);
  expect((await request(app).get('/api/visitors/not-an-id').set(auth())).status).toBe(404);
});
