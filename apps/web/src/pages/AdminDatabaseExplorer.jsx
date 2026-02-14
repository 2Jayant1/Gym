import React, { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Database, RefreshCcw, Search, Table2 } from 'lucide-react';
import api from '../services/api';

function classNames(...xs) {
  return xs.filter(Boolean).join(' ');
}

function formatCell(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'string') {
    if (value.length > 140) return `${value.slice(0, 140)}…`;
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export default function AdminDatabaseExplorer() {
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState(null);
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState('');
  const [loadingTables, setLoadingTables] = useState(true);
  const [loadingRows, setLoadingRows] = useState(false);
  const [error, setError] = useState(null);

  function loadTables() {
    setLoadingTables(true);
    setError(null);
    api
      .get('/api/admin/tables')
      .then((res) => {
        setTables(res.data || []);
        if (!selectedTable && res.data?.length) {
          setSelectedTable(res.data[0].table);
        }
      })
      .catch((e) => {
        console.error(e);
        setError(e?.response?.data?.error || e.message || 'Failed to load tables');
      })
      .finally(() => setLoadingTables(false));
  }

  function loadRows(tableName) {
    if (!tableName) return;
    setLoadingRows(true);
    setError(null);
    api
      .get(`/api/admin/table/${encodeURIComponent(tableName)}`)
      .then((res) => setRows(res.data || []))
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load rows');
      })
      .finally(() => setLoadingRows(false));
  }

  useEffect(() => {
    loadTables();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    loadRows(selectedTable);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTable]);

  const columns = useMemo(() => {
    const first = rows?.[0];
    if (!first) return [];
    return Object.keys(first);
  }, [rows]);

  const filteredRows = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) =>
      Object.values(r).some((v) =>
        String(v ?? '')
          .toLowerCase()
          .includes(q)
      )
    );
  }, [rows, filter]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 sm:gap-4">
        <div className="min-w-0">
          <h1 className="text-slate-900 text-2xl sm:text-3xl font-extrabold flex items-center gap-2">
            <Database className="h-6 w-6 sm:h-7 sm:w-7 text-slate-700 shrink-0" />
            Database Explorer
          </h1>
          <p className="text-slate-500 text-sm sm:text-base">
            Browse every seeded table via the admin API. Great for verification and demos.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button onClick={() => loadTables()} className="btn-ghost" title="Refresh table counts">
            <RefreshCcw className={classNames('h-4 w-4', loadingTables ? 'animate-spin' : '')} />
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="card border border-red-200/60">
          <div className="text-red-600 font-semibold">Something went wrong</div>
          <div className="text-slate-500 text-sm mt-1">{error}</div>
        </div>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <motion.div className="lg:col-span-4 card p-0 overflow-hidden" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }}>
          <div className="p-4 border-b border-slate-200 flex items-center justify-between">
            <div className="text-slate-700 font-bold flex items-center gap-2">
              <Table2 className="h-4 w-4 text-slate-600" />
              Tables
            </div>
            <div className="text-slate-500 text-xs">
              {loadingTables ? 'Loading…' : `${tables.length} total`}
            </div>
          </div>

          <div className="max-h-[520px] overflow-auto">
            {tables.map((t) => {
              const active = selectedTable === t.table;
              return (
                <button
                  key={t.table}
                  onClick={() => setSelectedTable(t.table)}
                  className={classNames(
                    'w-full text-left px-4 py-3 border-b border-slate-100 hover:bg-slate-50 transition-colors',
                    active ? 'bg-slate-50' : ''
                  )}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-slate-800 font-semibold">{t.table}</div>
                    <div className="text-slate-500 text-xs tabular-nums">
                      {t.error ? '—' : t.count}
                    </div>
                  </div>
                  {t.error ? <div className="text-red-600/80 text-xs mt-1">{t.error}</div> : null}
                </button>
              );
            })}
            {!loadingTables && tables.length === 0 ? (
              <div className="p-4 text-slate-500 text-sm">No tables available.</div>
            ) : null}
          </div>
        </motion.div>

        <motion.div
          key={selectedTable || 'none'}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18, ease: 'easeOut' }}
          className="lg:col-span-8 card p-0 overflow-hidden"
        >
          <div className="p-4 border-b border-slate-200 flex items-center justify-between gap-4">
            <div>
              <div className="text-slate-800 font-bold">{selectedTable || 'Select a table'}</div>
              <div className="text-slate-500 text-xs">Showing up to 200 rows (API limit)</div>
            </div>

            <div className="relative w-full max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter rows…"
                className="input pl-9"
              />
            </div>
          </div>

          <div className="p-4">
            {loadingRows ? <div className="text-slate-500">Loading rows…</div> : null}

            {!loadingRows && selectedTable && rows.length === 0 ? (
              <div className="text-slate-500 text-sm">No rows found.</div>
            ) : null}

            {!loadingRows && rows.length > 0 ? (
              <div className="overflow-auto rounded-xl border border-slate-200">
                <table className="min-w-full text-sm">
                  <thead className="bg-slate-50">
                    <tr>
                      {columns.map((c) => (
                        <th
                          key={c}
                          className="text-left text-slate-600 font-semibold px-3 py-2 whitespace-nowrap"
                        >
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.slice(0, 200).map((r, idx) => (
                      <tr
                        key={idx}
                        className="border-t border-slate-100 hover:bg-slate-50 transition-colors"
                      >
                        {columns.map((c) => (
                          <td key={c} className="px-3 py-2 text-slate-700 whitespace-nowrap">
                            {formatCell(r[c])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}

            {!loadingRows && rows.length > 0 && filteredRows.length !== rows.length ? (
              <div className="text-slate-500 text-xs mt-3">
                Filtered: {filteredRows.length} / {rows.length}
              </div>
            ) : null}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
