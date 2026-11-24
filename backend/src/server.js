require('dotenv').config();
const http = require('http');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const { Server } = require('socket.io');
const { createApp } = require('./app');
const { createMLClient } = require('./mlClient');
const { seedAdmin } = require('./auth');

const {
  PORT = 5000,
  MONGO_URI = 'mongodb://127.0.0.1:27017/visitrack',
  JWT_SECRET,
  GATE_KEY,
  ML_URL = 'http://127.0.0.1:5001',
  ML_SERVICE_KEY = '',
  ADMIN_EMAIL,
  ADMIN_PASSWORD,
  CORS_ORIGIN = '*',
  CHECKIN_COOLDOWN_SEC = 60,
} = process.env;

async function main() {
  if (!JWT_SECRET) throw new Error('Set JWT_SECRET in backend/.env (see .env.example)');

  await mongoose.connect(MONGO_URI);
  console.log(`MongoDB connected: ${MONGO_URI}`);
  if (await seedAdmin(ADMIN_EMAIL, ADMIN_PASSWORD)) console.log(`Created admin ${ADMIN_EMAIL}`);

  let io;
  const app = createApp({
    ml: createMLClient({ baseUrl: ML_URL, key: ML_SERVICE_KEY }),
    emit: (event, payload) => io?.emit(event, payload),
    jwtSecret: JWT_SECRET,
    gateKey: GATE_KEY,
    cooldownSec: Number(CHECKIN_COOLDOWN_SEC),
    corsOrigin: CORS_ORIGIN,
  });

  // Express must be the server's handler BEFORE Socket.IO attaches: Socket.IO then
  // takes /socket.io/* requests itself and passes everything else to Express.
  const server = http.createServer(app);
  // Socket.IO pushes new visits to every open dashboard, so admins see them instantly.
  io = new Server(server, { cors: { origin: CORS_ORIGIN } });
  io.use((socket, next) => {
    try {
      jwt.verify(socket.handshake.auth?.token, JWT_SECRET);
      next();
    } catch {
      next(new Error('unauthorized'));
    }
  });

  server.listen(PORT, () => console.log(`VisiTrack API on http://localhost:${PORT}`));
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
