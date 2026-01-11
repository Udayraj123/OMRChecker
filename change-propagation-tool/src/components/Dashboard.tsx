import { useState, useMemo } from 'react';
import type { FileMappingData, FileMapping } from '../types';

interface DashboardProps {
  mappingData: FileMappingData | null;
}

function Dashboard({ mappingData }: DashboardProps) {
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterPhase, setFilterPhase] = useState<string>('all');
  const [filterPriority, setFilterPriority] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredMappings = useMemo(() => {
    if (!mappingData) return [];

    return mappingData.mappings.filter((mapping) => {
      // Status filter
      if (filterStatus !== 'all' && mapping.status !== filterStatus) {
        return false;
      }

      // Phase filter
      if (filterPhase !== 'all' && String(mapping.phase) !== filterPhase) {
        return false;
      }

      // Priority filter
      if (filterPriority !== 'all' && mapping.priority !== filterPriority) {
        return false;
      }

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          mapping.python.toLowerCase().includes(query) ||
          (mapping.typescript?.toLowerCase().includes(query) ?? false)
        );
      }

      return true;
    });
  }, [mappingData, filterStatus, filterPhase, filterPriority, searchQuery]);

  const stats = mappingData?.statistics || {
    total: 0,
    synced: 0,
    partial: 0,
    not_started: 0,
  };

  const syncedPercentage = stats.total > 0 ? (stats.synced / stats.total) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Total Files"
          value={stats.total}
          color="blue"
          icon="📁"
        />
        <StatCard
          title="Synced"
          value={stats.synced}
          percentage={syncedPercentage}
          color="green"
          icon="✅"
        />
        <StatCard
          title="Partially Synced"
          value={stats.partial}
          color="yellow"
          icon="⚠️"
        />
        <StatCard
          title="Not Started"
          value={stats.not_started}
          color="red"
          icon="❌"
        />
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Filters</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search files..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Status
            </label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All</option>
              <option value="synced">Synced</option>
              <option value="partial">Partial</option>
              <option value="not_started">Not Started</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Phase
            </label>
            <select
              value={filterPhase}
              onChange={(e) => setFilterPhase(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All</option>
              <option value="1">Phase 1</option>
              <option value="2">Phase 2</option>
              <option value="future">Future</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Priority
            </label>
            <select
              value={filterPriority}
              onChange={(e) => setFilterPriority(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>
      </div>

      {/* File Mappings Grid */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">
            File Mappings ({filteredMappings.length})
          </h2>
        </div>
        <div className="divide-y divide-gray-200">
          {filteredMappings.length === 0 ? (
            <div className="px-6 py-12 text-center text-gray-500">
              No files match the current filters
            </div>
          ) : (
            filteredMappings.map((mapping, index) => (
              <FileMappingCard key={index} mapping={mapping} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: number;
  percentage?: number;
  color: 'blue' | 'green' | 'yellow' | 'red';
  icon: string;
}

function StatCard({ title, value, percentage, color, icon }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700',
    green: 'bg-green-50 text-green-700',
    yellow: 'bg-yellow-50 text-yellow-700',
    red: 'bg-red-50 text-red-700',
  };

  return (
    <div className={`rounded-lg shadow p-6 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium opacity-80">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
          {percentage !== undefined && (
            <p className="text-sm mt-1">{percentage.toFixed(1)}%</p>
          )}
        </div>
        <div className="text-4xl opacity-50">{icon}</div>
      </div>
    </div>
  );
}

interface FileMappingCardProps {
  mapping: FileMapping;
}

function FileMappingCard({ mapping }: FileMappingCardProps) {
  const statusColors = {
    synced: 'bg-green-100 text-green-800',
    partial: 'bg-yellow-100 text-yellow-800',
    not_started: 'bg-red-100 text-red-800',
  };

  const priorityColors = {
    high: 'bg-red-100 text-red-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-gray-100 text-gray-800',
  };

  return (
    <div className="px-6 py-4 hover:bg-gray-50 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                statusColors[mapping.status]
              }`}
            >
              {mapping.status === 'synced' && '✅'}
              {mapping.status === 'partial' && '⚠️'}
              {mapping.status === 'not_started' && '❌'}
              <span className="ml-1">{mapping.status.replace('_', ' ')}</span>
            </span>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                priorityColors[mapping.priority]
              }`}
            >
              {mapping.priority}
            </span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              Phase {mapping.phase}
            </span>
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-gray-500">Python:</span>
              <code className="text-sm text-gray-900 font-mono">
                {mapping.python}
              </code>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-gray-500">TypeScript:</span>
              <code className="text-sm text-gray-700 font-mono">
                {mapping.typescript || 'Not mapped'}
              </code>
            </div>
          </div>
          {mapping.notes && (
            <p className="mt-2 text-sm text-gray-600 italic">{mapping.notes}</p>
          )}
        </div>
        <div className="ml-4 flex-shrink-0">
          <button className="px-3 py-1 text-sm font-medium text-primary-600 hover:text-primary-700 hover:bg-primary-50 rounded transition-colors">
            View Details
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;

