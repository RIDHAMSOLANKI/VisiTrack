const mongoose = require('mongoose');

const adminSchema = new mongoose.Schema(
  {
    email: { type: String, required: true, unique: true, lowercase: true, trim: true },
    passwordHash: { type: String, required: true },
    name: { type: String, default: 'Admin' },
  },
  { timestamps: true }
);

const visitorSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, trim: true, maxlength: 100 },
    phone: { type: String, trim: true, maxlength: 20 },
    email: { type: String, trim: true, lowercase: true, maxlength: 120 },
    purpose: { type: String, trim: true, maxlength: 200 },
    host: { type: String, trim: true, maxlength: 100 }, // person being visited
    faceCount: { type: Number, default: 0 }, // embeddings stored in the ML service
    active: { type: Boolean, default: true },
  },
  { timestamps: true }
);
visitorSchema.index({ name: 'text', phone: 'text', email: 'text' });

const visitSchema = new mongoose.Schema(
  {
    visitor: { type: mongoose.Schema.Types.ObjectId, ref: 'Visitor', default: null },
    visitorName: { type: String, default: 'Unknown' }, // copy, so history survives visitor deletion
    status: { type: String, enum: ['recognized', 'unknown'], required: true },
    gate: { type: String, default: 'main', trim: true, maxlength: 50 },
    score: { type: Number, default: null }, // average similarity from the model
    matchRatio: { type: Number, default: null }, // share of stored photos that matched
    checkInAt: { type: Date, default: Date.now },
    checkOutAt: { type: Date, default: null },
  },
  { timestamps: true }
);
// Fast "latest visits", per-visitor history and filters.
visitSchema.index({ checkInAt: -1 });
visitSchema.index({ visitor: 1, checkInAt: -1 });
visitSchema.index({ status: 1, checkInAt: -1 });

module.exports = {
  Admin: mongoose.model('Admin', adminSchema),
  Visitor: mongoose.model('Visitor', visitorSchema),
  Visit: mongoose.model('Visit', visitSchema),
};
