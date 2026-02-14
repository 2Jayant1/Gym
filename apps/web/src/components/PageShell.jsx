import React from 'react';
import { motion } from 'framer-motion';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export default function PageShell({ title, subtitle, right, children }) {
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-5 sm:space-y-6">
      <motion.div
        variants={itemVariants}
        className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 sm:gap-4"
      >
        <div className="min-w-0">
          <h1 className="text-slate-900 text-2xl sm:text-3xl font-extrabold truncate bg-gradient-to-r from-slate-900 via-slate-800 to-slate-700 bg-clip-text">
            {title}
          </h1>
          {subtitle ? (
            <motion.p
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.15, duration: 0.4 }}
              className="text-slate-500 text-sm sm:text-base mt-0.5"
            >
              {subtitle}
            </motion.p>
          ) : null}
        </div>
        {right ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.3 }}
            className="flex items-center gap-2 flex-wrap shrink-0"
          >
            {right}
          </motion.div>
        ) : null}
      </motion.div>

      <motion.div variants={itemVariants} className="space-y-5 sm:space-y-6">
        {children}
      </motion.div>
    </motion.div>
  );
}
