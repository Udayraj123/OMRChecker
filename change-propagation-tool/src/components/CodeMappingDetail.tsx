import { useState } from 'react';
import type { FileMapping, ClassMapping, MethodMapping, FunctionMapping } from '../types';

interface CodeMappingDetailProps {
  mapping: FileMapping;
  onClose: () => void;
}

function CodeMappingDetail({ mapping, onClose }: CodeMappingDetailProps) {
  const [expandedClasses, setExpandedClasses] = useState<Set<string>>(new Set());

  const toggleClass = (className: string) => {
    const newExpanded = new Set(expandedClasses);
    if (newExpanded.has(className)) {
      newExpanded.delete(className);
    } else {
      newExpanded.add(className);
    }
    setExpandedClasses(newExpanded);
  };

  const hasCodeMappings = (mapping.classes && mapping.classes.length > 0) ||
                          (mapping.functions && mapping.functions.length > 0);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">Code Mapping Details</h2>
              <p className="text-blue-100 text-sm mt-1">
                {mapping.python.split('/').pop()}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:bg-blue-500 rounded-full p-2 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* File Info */}
          <div className="space-y-3">
            <InfoRow label="Python File" value={mapping.python} mono />
            <InfoRow
              label="TypeScript File"
              value={mapping.typescript || 'Not mapped'}
              mono
            />
            {mapping.testFile && (
              <InfoRow label="Test File" value={mapping.testFile} mono />
            )}
            <div className="flex gap-4">
              <InfoRow
                label="Status"
                value={
                  <StatusBadge status={mapping.status} />
                }
              />
              <InfoRow
                label="Phase"
                value={`Phase ${mapping.phase}`}
              />
              <InfoRow
                label="Priority"
                value={<PriorityBadge priority={mapping.priority} />}
              />
            </div>
            {mapping.lastTypescriptChange && (
              <InfoRow
                label="Last Updated"
                value={new Date(mapping.lastTypescriptChange).toLocaleDateString()}
              />
            )}
          </div>

          {mapping.notes && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-blue-900 mb-2">📝 Notes</h4>
              <p className="text-sm text-blue-800 whitespace-pre-wrap">{mapping.notes}</p>
            </div>
          )}

          {/* Classes */}
          {mapping.classes && mapping.classes.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <span className="text-blue-600">🏛️</span>
                Classes ({mapping.classes.length})
              </h3>
              <div className="space-y-3">
                {mapping.classes.map((cls, idx) => (
                  <ClassCard
                    key={idx}
                    classMapping={cls}
                    isExpanded={expandedClasses.has(cls.python)}
                    onToggle={() => toggleClass(cls.python)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Functions */}
          {mapping.functions && mapping.functions.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <span className="text-purple-600">⚡</span>
                Functions ({mapping.functions.length})
              </h3>
              <div className="space-y-2">
                {mapping.functions.map((func, idx) => (
                  typeof func === 'string' ? (
                    <div key={idx} className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                      <code className="text-sm text-gray-800 font-mono">{func}</code>
                    </div>
                  ) : (
                    <FunctionCard key={idx} functionMapping={func} />
                  )
                ))}
              </div>
            </div>
          )}

          {!hasCodeMappings && (
            <div className="text-center py-12 text-gray-500">
              <p className="text-lg mb-2">No detailed code mappings available</p>
              <p className="text-sm">This file may contain utilities or simple exports</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

interface InfoRowProps {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}

function InfoRow({ label, value, mono }: InfoRowProps) {
  return (
    <div className="flex items-start gap-3">
      <span className="text-sm font-medium text-gray-500 min-w-32">{label}:</span>
      <span className={`text-sm text-gray-900 ${mono ? 'font-mono bg-gray-100 px-2 py-1 rounded' : ''}`}>
        {value}
      </span>
    </div>
  );
}

interface StatusBadgeProps {
  status: string;
}

function StatusBadge({ status }: StatusBadgeProps) {
  const colors = {
    synced: 'bg-green-100 text-green-800',
    partial: 'bg-yellow-100 text-yellow-800',
    not_started: 'bg-red-100 text-red-800',
  };

  const icons = {
    synced: '✅',
    partial: '⚠️',
    not_started: '❌',
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[status as keyof typeof colors]}`}>
      {icons[status as keyof typeof icons]} {status.replace('_', ' ')}
    </span>
  );
}

interface PriorityBadgeProps {
  priority: string;
}

function PriorityBadge({ priority }: PriorityBadgeProps) {
  const colors = {
    high: 'bg-red-100 text-red-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-gray-100 text-gray-800',
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[priority as keyof typeof colors]}`}>
      {priority}
    </span>
  );
}

interface ClassCardProps {
  classMapping: ClassMapping;
  isExpanded: boolean;
  onToggle: () => void;
}

function ClassCard({ classMapping, isExpanded, onToggle }: ClassCardProps) {
  const methodCount = classMapping.methods?.length || 0;
  const syncedMethodCount = classMapping.methods?.filter(m => m.synced).length || 0;
  const methodProgress = methodCount > 0 ? (syncedMethodCount / methodCount) * 100 : 0;

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <div
        className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 cursor-pointer hover:from-blue-100 hover:to-indigo-100 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <code className="text-base font-semibold text-gray-900 font-mono">
                {classMapping.python}
              </code>
              {classMapping.synced && (
                <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full font-medium">
                  ✓ Synced
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span>→</span>
              <code className="font-mono text-gray-700">{classMapping.typescript}</code>
            </div>
            {methodCount > 0 && (
              <div className="mt-2 flex items-center gap-3">
                <span className="text-xs text-gray-600">
                  {syncedMethodCount}/{methodCount} methods synced
                </span>
                <div className="flex-1 max-w-xs bg-gray-200 rounded-full h-1.5">
                  <div
                    className="bg-green-500 h-full rounded-full transition-all"
                    style={{ width: `${methodProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
          <svg
            className={`w-5 h-5 text-gray-500 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
        {classMapping.note && (
          <p className="mt-2 text-xs text-gray-600 italic">{classMapping.note}</p>
        )}
      </div>

      {isExpanded && classMapping.methods && classMapping.methods.length > 0 && (
        <div className="bg-white p-4 border-t border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Methods:</h4>
          <div className="space-y-2">
            {classMapping.methods.map((method, idx) => (
              <MethodCard key={idx} method={method} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MethodCardProps {
  method: MethodMapping;
}

function MethodCard({ method }: MethodCardProps) {
  return (
    <div className={`border rounded p-3 ${method.synced ? 'border-green-200 bg-green-50' : 'border-gray-200 bg-gray-50'}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <code className="text-sm font-mono text-gray-800">{method.python}</code>
            <span className="text-gray-400">→</span>
            <code className="text-sm font-mono text-gray-700">{method.typescript}</code>
          </div>
          {method.notes && (
            <p className="text-xs text-gray-600 italic mt-1">{method.notes}</p>
          )}
        </div>
        {method.synced ? (
          <span className="text-green-600 text-sm">✓</span>
        ) : (
          <span className="text-gray-400 text-sm">○</span>
        )}
      </div>
    </div>
  );
}

interface FunctionCardProps {
  functionMapping: FunctionMapping;
}

function FunctionCard({ functionMapping }: FunctionCardProps) {
  return (
    <div className={`border rounded-lg p-3 ${functionMapping.synced ? 'border-green-200 bg-green-50' : 'border-gray-200 bg-gray-50'}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <code className="text-sm font-mono text-gray-800">{functionMapping.python}</code>
            <span className="text-gray-400">→</span>
            <code className="text-sm font-mono text-gray-700">{functionMapping.typescript}</code>
          </div>
          {functionMapping.notes && (
            <p className="text-xs text-gray-600 italic mt-1">{functionMapping.notes}</p>
          )}
        </div>
        {functionMapping.synced ? (
          <span className="text-green-600 text-sm">✓</span>
        ) : (
          <span className="text-gray-400 text-sm">○</span>
        )}
      </div>
    </div>
  );
}

export default CodeMappingDetail;

