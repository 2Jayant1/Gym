import React from 'react';

export default function PlaceholderPage({ title, description }) {
  return (
    <div className="card">
      <h1 className="text-slate-900 text-3xl font-extrabold">{title}</h1>
      <p className="text-slate-500 mt-2">{description}</p>
      <div className="mt-6 text-slate-500 text-sm">
        This section is scaffolded for you. Tell me what features you want here and I’ll implement them next.
      </div>
    </div>
  );
}
