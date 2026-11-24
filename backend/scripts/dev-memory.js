// Run the API with a throwaway in-memory MongoDB (no MongoDB install needed).
// Data is lost when the process stops.   npm run dev:memory
const { MongoMemoryServer } = require('mongodb-memory-server');

(async () => {
  const mongo = await MongoMemoryServer.create();
  process.env.MONGO_URI = mongo.getUri('visitrack');
  process.env.JWT_SECRET ||= 'dev-only-secret';
  process.env.GATE_KEY ||= 'dev-gate-key';
  process.env.ADMIN_EMAIL ||= 'admin@visitrack.local';
  process.env.ADMIN_PASSWORD ||= 'admin123';
  console.log(`In-memory MongoDB at ${process.env.MONGO_URI}`);
  console.log(`Login: ${process.env.ADMIN_EMAIL} / ${process.env.ADMIN_PASSWORD}   Gate key: ${process.env.GATE_KEY}`);
  require('../src/server');
})();
