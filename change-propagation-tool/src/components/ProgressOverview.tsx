import type { FileMappingData } from '../types';

interface ProgressOverviewProps {
  mappingData: FileMappingData | null;
}

function ProgressOverview({ mappingData }: ProgressOverviewProps) {
  if (!mappingData) return null;

  const stats = mappingData.statistics;
  const metadata = mappingData.metadata;

  const phases = [
    {
      id: 'phase7',
      name: 'Phase 7',
      title: metadata?.phase7_summary || 'Phase 7',
      completed: metadata?.phase7_completed,
      improvements: metadata?.phase7_improvements
    },
    {
      id: 'phase8',
      name: 'Phase 8',
      title: metadata?.phase8_summary || 'Phase 8',
      completed: metadata?.phase8_completed,
      improvements: metadata?.phase8_improvements
    },
    {
      id: 'phase9',
      name: 'Phase 9',
      title: metadata?.phase9_summary || 'Phase 9',
      completed: metadata?.phase9_completed,
      improvements: metadata?.phase9_improvements
    },
  ];

  const overallProgress = stats ? (stats.synced / stats.total) * 100 : 0;
  const phase1Progress = stats && stats.phase1 ? (stats.synced / stats.phase1) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Overall Progress */}
      <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 text-white">
        <h2 className="text-2xl font-bold mb-4">Overall Porting Progress</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-blue-100 text-sm mb-1">Total Progress</p>
            <div className="flex items-end gap-2">
              <p className="text-5xl font-bold">{overallProgress.toFixed(0)}%</p>
              <p className="text-blue-100 text-lg pb-2">{stats?.synced}/{stats?.total} files</p>
            </div>
            <div className="mt-3 bg-blue-400 rounded-full h-3 overflow-hidden">
              <div
                className="bg-white h-full rounded-full transition-all duration-500"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>

          <div>
            <p className="text-blue-100 text-sm mb-1">Phase 1 (Core)</p>
            <div className="flex items-end gap-2">
              <p className="text-5xl font-bold">{phase1Progress.toFixed(0)}%</p>
              <p className="text-blue-100 text-lg pb-2">{stats?.phase1} files</p>
            </div>
            <div className="mt-3 bg-blue-400 rounded-full h-3 overflow-hidden">
              <div
                className="bg-green-300 h-full rounded-full transition-all duration-500"
                style={{ width: `${phase1Progress}%` }}
              />
            </div>
          </div>

          <div>
            <p className="text-blue-100 text-sm mb-1">Status Breakdown</p>
            <div className="space-y-2 mt-2">
              <div className="flex justify-between items-center">
                <span className="text-sm">✅ Synced</span>
                <span className="font-semibold">{stats?.synced}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">⚠️ Partial</span>
                <span className="font-semibold">{stats?.partial}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">❌ Not Started</span>
                <span className="font-semibold">{stats?.not_started}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Phases */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Development Phases</h3>
        <div className="space-y-4">
          {phases.map((phase) => phase.completed && (
            <PhaseCard key={phase.id} phase={phase} />
          ))}
        </div>
      </div>

      {/* Phase Distribution */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-sm font-medium text-gray-500 mb-2">Phase 1: Core</h4>
          <p className="text-3xl font-bold text-gray-900">{stats?.phase1 || 0}</p>
          <p className="text-sm text-gray-600 mt-1">Essential processors</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-sm font-medium text-gray-500 mb-2">Phase 2: Advanced</h4>
          <p className="text-3xl font-bold text-gray-900">{stats?.phase2 || 0}</p>
          <p className="text-sm text-gray-600 mt-1">Advanced features</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-sm font-medium text-gray-500 mb-2">Future</h4>
          <p className="text-3xl font-bold text-gray-900">{stats?.future || 0}</p>
          <p className="text-sm text-gray-600 mt-1">ML & OCR features</p>
        </div>
      </div>
    </div>
  );
}

interface PhaseCardProps {
  phase: {
    id: string;
    name: string;
    title: string;
    completed?: string;
    improvements?: Record<string, boolean>;
  };
}

function PhaseCard({ phase }: PhaseCardProps) {
  const improvementCount = phase.improvements
    ? Object.values(phase.improvements).filter(Boolean).length
    : 0;

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-md transition-all">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h4 className="font-semibold text-gray-900">{phase.name}</h4>
          <p className="text-sm text-gray-600 mt-1">{phase.title}</p>
        </div>
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          ✓ Complete
        </span>
      </div>

      {phase.completed && (
        <p className="text-xs text-gray-500 mt-2">
          Completed: {new Date(phase.completed).toLocaleDateString()}
        </p>
      )}

      {phase.improvements && improvementCount > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <p className="text-xs font-medium text-gray-700 mb-2">
            Key Improvements ({improvementCount}):
          </p>
          <div className="grid grid-cols-1 gap-1">
            {Object.entries(phase.improvements).map(([key, value]) =>
              value && (
                <div key={key} className="flex items-center gap-2 text-xs text-gray-600">
                  <span className="text-green-500">✓</span>
                  <span>{key.replace(/_/g, ' ')}</span>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ProgressOverview;

