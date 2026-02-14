import React from 'react';
import { AlertTriangle, RefreshCcw } from 'lucide-react';

/**
 * Global error boundary — catches any unhandled React render error
 * and shows a friendly recovery UI instead of a white screen.
 *
 * Usage:  <ErrorBoundary> <App /> </ErrorBoundary>
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Log to console (replace with a real error tracker like Sentry in production)
    console.error('[ErrorBoundary]', error, info?.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-6">
        <div className="max-w-md w-full text-center space-y-6">
          {/* Icon */}
          <div className="mx-auto h-20 w-20 rounded-2xl bg-red-100 flex items-center justify-center">
            <AlertTriangle className="h-10 w-10 text-red-500" />
          </div>

          {/* Copy */}
          <div>
            <h1 className="text-slate-900 text-2xl font-bold">Something went wrong</h1>
            <p className="text-slate-500 mt-2 text-sm leading-relaxed">
              An unexpected error occurred. You can try again or reload the page.
            </p>
          </div>

          {/* Error detail (dev only) */}
          {import.meta.env.DEV && this.state.error && (
            <pre className="text-left text-xs text-red-600 bg-red-50 border border-red-200 rounded-xl p-4 overflow-auto max-h-40">
              {this.state.error.message || String(this.state.error)}
            </pre>
          )}

          {/* Actions */}
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={this.handleReload}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
            >
              <RefreshCcw className="h-4 w-4" /> Reload Page
            </button>
          </div>
        </div>
      </div>
    );
  }
}
