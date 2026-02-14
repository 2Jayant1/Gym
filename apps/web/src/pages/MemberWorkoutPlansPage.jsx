import React, { useEffect, useState } from 'react';
import { Dumbbell, RefreshCcw, ChevronDown, ChevronUp } from 'lucide-react';
import PageShell from '../components/PageShell';
import api from '../services/api';
import { motion } from 'framer-motion';

const dayNames = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export default function MemberWorkoutPlansPage() {
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(null);

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/me/workout-plans')
      .then((res) => setPlans(res.data || []))
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load workout plans'))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <PageShell
      title="Workout Plans"
      subtitle="Plans assigned to you by your trainer."
      right={
        <button onClick={load} className="btn-ghost" title="Refresh">
          <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
          Refresh
        </button>
      }
    >
      {error && <div className="card text-red-600 border border-red-200/60">{error}</div>}
      {loading && <div className="card text-slate-500">Loading workout plans…</div>}

      {!loading && !error && plans.length === 0 && (
        <div className="card text-center py-12 text-slate-500">
          <Dumbbell className="h-12 w-12 mx-auto mb-3 text-slate-300" />
          <div className="text-lg font-semibold text-slate-700">No workout plans yet</div>
          <div className="text-sm">Your trainer will assign plans once you're set up.</div>
        </div>
      )}

      {!loading &&
        !error &&
        plans.map((plan, i) => {
          const isExpanded = expanded === plan._id;
          // Group exercises by day
          const exercisesByDay = {};
          (plan.exercises || []).forEach((we) => {
            const day = we.dayOfWeek || 0;
            if (!exercisesByDay[day]) exercisesByDay[day] = [];
            exercisesByDay[day].push(we);
          });

          return (
            <motion.div key={plan._id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }} whileHover={{ y: -2 }} className="card p-0 overflow-hidden">
              <button
                onClick={() => setExpanded(isExpanded ? null : plan._id)}
                className="w-full p-4 flex items-center justify-between hover:bg-slate-50 transition-colors"
              >
                <div className="text-left">
                  <div className="text-slate-900 font-bold text-lg">{plan.name}</div>
                  <div className="text-slate-500 text-sm">
                    {plan.goal || '—'} • {plan.difficultyLevel || 'All levels'}
                    {plan.trainer_name ? ` • Coach: ${plan.trainer_name}` : ''}
                  </div>
                  {plan.description && <div className="text-slate-500 text-sm mt-1">{plan.description}</div>}
                  <div className="text-slate-400 text-xs mt-1">
                    {plan.startDate ? new Date(plan.startDate).toLocaleDateString() : '—'} →{' '}
                    {plan.endDate ? new Date(plan.endDate).toLocaleDateString() : 'ongoing'}
                  </div>
                </div>
                {isExpanded ? <ChevronUp className="h-5 w-5 text-slate-400" /> : <ChevronDown className="h-5 w-5 text-slate-400" />}
              </button>

              {isExpanded && (
                <div className="border-t border-slate-200 p-4 space-y-4">
                  {Object.entries(exercisesByDay)
                    .sort(([a], [b]) => Number(a) - Number(b))
                    .map(([day, exs]) => (
                      <div key={day}>
                        <div className="text-slate-700 font-semibold mb-2">
                          {dayNames[Number(day)] || `Day ${day}`}
                        </div>
                        <div className="space-y-2">
                          {exs
                            .sort((a, b) => (a.sortOrder || 0) - (b.sortOrder || 0))
                            .map((we, i) => {
                              const ex = we.exercise || {};
                              return (
                                <div key={we._id || i} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                                  <div>
                                    <div className="text-slate-800 font-medium text-sm">{ex.name || 'Exercise'}</div>
                                    <div className="text-slate-500 text-xs">
                                      {ex.muscleGroup || ''} {ex.difficultyLevel ? `• ${ex.difficultyLevel}` : ''}
                                    </div>
                                  </div>
                                  <div className="text-right text-sm text-slate-600">
                                    {we.sets && we.reps && <span>{we.sets}×{we.reps}</span>}
                                    {we.weightKg ? <span className="ml-2">{we.weightKg}kg</span> : null}
                                    {we.durationMinutes ? <span className="ml-2">{we.durationMinutes}min</span> : null}
                                    {we.restSeconds ? <span className="ml-2 text-slate-400">rest {we.restSeconds}s</span> : null}
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    ))}
                  {Object.keys(exercisesByDay).length === 0 && (
                    <div className="text-slate-500 text-sm">No exercises assigned yet.</div>
                  )}
                </div>
              )}
            </motion.div>
          );
        })}
    </PageShell>
  );
}
