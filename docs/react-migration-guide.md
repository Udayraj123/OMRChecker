# React Migration Guide for Workflow Visualization

## Overview

This guide explains how to migrate the workflow visualization from vanilla JavaScript to a modern React-based application with better performance, maintainability, and user experience.

## Why Migrate to React?

### Current Issues (Vanilla JS)
- ❌ 680+ lines in single HTML file
- ❌ Hard to maintain and test
- ❌ No component reusability
- ❌ CDN dependencies (can fail offline)
- ❌ Limited interactivity and animations
- ❌ No type safety
- ❌ Manual DOM manipulation
- ❌ Difficult to add new features

### Benefits of React
- ✅ Component-based architecture
- ✅ TypeScript for type safety
- ✅ Better performance (Virtual DOM)
- ✅ Hot module reload for fast development
- ✅ Easy to test (Jest + React Testing Library)
- ✅ Rich ecosystem of libraries
- ✅ Better state management
- ✅ Modern build tools (Vite)
- ✅ Easy to extend and maintain

## Recommended Tech Stack

```
Core:
- React 18+ (UI framework)
- TypeScript (type safety)
- Vite (build tool, ~10x faster than CRA)

Visualization:
- React Flow (better than vis.js, built for React)
  or
- Cytoscape.js with React wrapper

UI/Styling:
- Tailwind CSS (utility-first CSS)
- Radix UI (headless components)
- Lucide React (icons)

State Management:
- Zustand (simple, powerful state)
  or
- Jotai (atomic state)

Data Loading:
- TanStack Query (React Query v5)

Animation:
- Framer Motion (smooth animations)

Testing:
- Vitest (faster than Jest)
- React Testing Library
- Playwright (E2E)
```

## Project Structure

```
workflow-viz-app/
├── public/
│   └── sample-sessions/          # Sample JSON files
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── MainLayout.tsx
│   │   │   └── Sidebar.tsx
│   │   ├── workflow/
│   │   │   ├── WorkflowGraph.tsx      # Main graph component
│   │   │   ├── GraphNode.tsx          # Custom node
│   │   │   ├── GraphEdge.tsx          # Custom edge
│   │   │   └── MiniMap.tsx            # Overview map
│   │   ├── viewer/
│   │   │   ├── ImageViewer.tsx        # Image display
│   │   │   ├── ImageComparison.tsx    # Before/after
│   │   │   └── ZoomControls.tsx       # Zoom in/out
│   │   ├── playback/
│   │   │   ├── Timeline.tsx           # Timeline scrubber
│   │   │   ├── PlaybackControls.tsx   # Play/pause/speed
│   │   │   └── ProgressBar.tsx        # Progress indicator
│   │   ├── metadata/
│   │   │   ├── MetadataPanel.tsx      # Processor details
│   │   │   ├── StatsCard.tsx          # Statistics
│   │   │   └── ErrorDisplay.tsx       # Error messages
│   │   └── common/
│   │       ├── Button.tsx
│   │       ├── Slider.tsx
│   │       └── Modal.tsx
│   ├── hooks/
│   │   ├── useWorkflowSession.ts      # Load session data
│   │   ├── usePlayback.ts             # Animation state
│   │   ├── useKeyboardShortcuts.ts    # Keyboard controls
│   │   └── useLocalStorage.ts         # Persist settings
│   ├── stores/
│   │   ├── sessionStore.ts            # Session state (Zustand)
│   │   ├── playbackStore.ts           # Playback state
│   │   └── uiStore.ts                 # UI preferences
│   ├── types/
│   │   ├── workflow.ts                # Workflow types
│   │   ├── session.ts                 # Session types
│   │   └── graph.ts                   # Graph types
│   ├── utils/
│   │   ├── imageUtils.ts              # Image processing
│   │   ├── graphUtils.ts              # Graph calculations
│   │   └── sessionLoader.ts           # Load JSON
│   ├── App.tsx                        # Main app
│   └── main.tsx                       # Entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .env.example
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── README.md
```

## Step-by-Step Migration

### Phase 1: Setup (30 minutes)

```bash
# Create React app with Vite
npm create vite@latest workflow-viz-app -- --template react-ts
cd workflow-viz-app

# Install dependencies
npm install

# Install additional packages
npm install react-flow-renderer zustand @tanstack/react-query
npm install framer-motion lucide-react
npm install -D tailwindcss postcss autoprefixer
npm install -D vitest @testing-library/react @testing-library/jest-dom

# Setup Tailwind
npx tailwindcss init -p
```

### Phase 2: Core Types (15 minutes)

Create `src/types/workflow.ts`:

```typescript
export interface ProcessorState {
  name: string;
  order: number;
  timestamp: string;
  duration_ms: number;
  image_shape: [number, number] | [number, number, number];
  gray_image_base64: string | null;
  colored_image_base64: string | null;
  metadata: Record<string, any>;
  success: boolean;
  error_message: string | null;
}

export interface GraphNode {
  id: string;
  label: string;
  metadata: {
    type: 'input' | 'processor' | 'output';
    order?: number;
  };
}

export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
}

export interface WorkflowSession {
  session_id: string;
  file_path: string;
  template_name: string;
  start_time: string;
  end_time: string | null;
  total_duration_ms: number | null;
  processor_states: ProcessorState[];
  graph: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  config: Record<string, any>;
  metadata: Record<string, any>;
}
```

### Phase 3: State Management (20 minutes)

Create `src/stores/sessionStore.ts`:

```typescript
import { create } from 'zustand';
import type { WorkflowSession, ProcessorState } from '../types/workflow';

interface SessionState {
  session: WorkflowSession | null;
  currentStep: number;
  isPlaying: boolean;
  playbackSpeed: number;
  showColored: boolean;

  // Actions
  loadSession: (session: WorkflowSession) => void;
  setCurrentStep: (step: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  toggleColorMode: () => void;
  nextStep: () => void;
  previousStep: () => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  session: null,
  currentStep: 0,
  isPlaying: false,
  playbackSpeed: 1000, // ms
  showColored: false,

  loadSession: (session) => set({ session, currentStep: 0 }),

  setCurrentStep: (step) => {
    const { session } = get();
    if (session && step >= 0 && step < session.processor_states.length) {
      set({ currentStep: step });
    }
  },

  setIsPlaying: (playing) => set({ isPlaying: playing }),

  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  toggleColorMode: () => set((state) => ({
    showColored: !state.showColored
  })),

  nextStep: () => {
    const { currentStep, session } = get();
    if (session && currentStep < session.processor_states.length - 1) {
      set({ currentStep: currentStep + 1 });
    }
  },

  previousStep: () => {
    const { currentStep } = get();
    if (currentStep > 0) {
      set({ currentStep: currentStep - 1 });
    }
  },

  reset: () => set({ currentStep: 0, isPlaying: false }),
}));
```

### Phase 4: Workflow Graph Component (45 minutes)

Create `src/components/workflow/WorkflowGraph.tsx`:

```typescript
import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MiniMap,
  BackgroundVariant,
} from 'react-flow-renderer';
import { useSessionStore } from '../../stores/sessionStore';

const nodeTypes = {
  // Define custom node types if needed
};

export const WorkflowGraph: React.FC = () => {
  const session = useSessionStore((state) => state.session);
  const currentStep = useSessionStore((state) => state.currentStep);
  const setCurrentStep = useSessionStore((state) => state.setCurrentStep);

  // Convert graph data to React Flow format
  const initialNodes: Node[] = useMemo(() => {
    if (!session) return [];

    return session.graph.nodes.map((node) => ({
      id: node.id,
      data: {
        label: node.label,
        metadata: node.metadata
      },
      position: { x: 0, y: 0 }, // Will be auto-layouted
      type: node.metadata.type === 'processor' ? 'default' : 'ellipse',
      style: {
        background: node.metadata.type === 'input' ? '#3498db' :
                   node.metadata.type === 'output' ? '#27ae60' : '#ecf0f1',
        color: node.metadata.type === 'processor' ? '#333' : 'white',
        border: '2px solid #555',
      },
    }));
  }, [session]);

  const initialEdges: Edge[] = useMemo(() => {
    if (!session) return [];

    return session.graph.edges.map((edge, idx) => ({
      id: `edge-${idx}`,
      source: edge.from,
      target: edge.to,
      animated: true,
      style: { stroke: '#bdc3c7' },
    }));
  }, [session]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Highlight current node
  React.useEffect(() => {
    if (!session) return;

    const currentState = session.processor_states[currentStep];
    const currentNodeId = currentState.name === 'Input' ? 'input' :
                         `processor_${currentState.order - 1}`;

    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        style: {
          ...node.style,
          border: node.id === currentNodeId ? '3px solid #e74c3c' : '2px solid #555',
          boxShadow: node.id === currentNodeId ? '0 0 10px #e74c3c' : 'none',
        },
      }))
    );
  }, [currentStep, session, setNodes]);

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    const processorIndex = node.data.metadata.order;
    if (processorIndex !== undefined && processorIndex >= 0) {
      setCurrentStep(processorIndex + 1);
    } else if (node.data.metadata.type === 'input') {
      setCurrentStep(0);
    }
  }, [setCurrentStep]);

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No session loaded
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      nodeTypes={nodeTypes}
      fitView
    >
      <Background variant={BackgroundVariant.Dots} />
      <Controls />
      <MiniMap />
    </ReactFlow>
  );
};
```

### Phase 5: Image Viewer Component (30 minutes)

Create `src/components/viewer/ImageViewer.tsx`:

```typescript
import React from 'react';
import { useSessionStore } from '../../stores/sessionStore';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

export const ImageViewer: React.FC = () => {
  const session = useSessionStore((state) => state.session);
  const currentStep = useSessionStore((state) => state.currentStep);
  const showColored = useSessionStore((state) => state.showColored);
  const toggleColorMode = useSessionStore((state) => state.toggleColorMode);

  const [zoom, setZoom] = React.useState(1);

  if (!session || !session.processor_states[currentStep]) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No image available
      </div>
    );
  }

  const state = session.processor_states[currentStep];
  const imageData = showColored && state.colored_image_base64
    ? state.colored_image_base64
    : state.gray_image_base64;

  if (!imageData) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No image data for this processor
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="text-lg font-semibold">{state.name}</h2>
        <div className="flex gap-2">
          {state.colored_image_base64 && (
            <button
              onClick={toggleColorMode}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              {showColored ? 'Show Grayscale' : 'Show Colored'}
            </button>
          )}
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="p-2 hover:bg-gray-100 rounded"
          >
            <ZoomOut size={20} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
            className="p-2 hover:bg-gray-100 rounded"
          >
            <ZoomIn size={20} />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-2 hover:bg-gray-100 rounded"
          >
            <Maximize2 size={20} />
          </button>
        </div>
      </div>

      {/* Image Display */}
      <div className="flex-1 overflow-auto bg-gray-50 p-4">
        <div className="flex items-center justify-center min-h-full">
          <img
            src={`data:image/jpeg;base64,${imageData}`}
            alt={state.name}
            style={{ transform: `scale(${zoom})` }}
            className="transition-transform shadow-lg"
          />
        </div>
      </div>

      {/* Metadata */}
      <div className="p-4 border-t bg-white">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-semibold">Duration:</span>{' '}
            {state.duration_ms.toFixed(2)} ms
          </div>
          <div>
            <span className="font-semibold">Size:</span>{' '}
            {state.image_shape.join(' × ')}
          </div>
          <div>
            <span className="font-semibold">Status:</span>{' '}
            <span className={state.success ? 'text-green-600' : 'text-red-600'}>
              {state.success ? 'Success' : 'Failed'}
            </span>
          </div>
          <div>
            <span className="font-semibold">Order:</span> {state.order}
          </div>
        </div>
      </div>
    </div>
  );
};
```

### Phase 6: Playback Controls (25 minutes)

Create `src/components/playback/PlaybackControls.tsx`:

```typescript
import React from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { useSessionStore } from '../../stores/sessionStore';

export const PlaybackControls: React.FC = () => {
  const session = useSessionStore((state) => state.session);
  const currentStep = useSessionStore((state) => state.currentStep);
  const isPlaying = useSessionStore((state) => state.isPlaying);
  const playbackSpeed = useSessionStore((state) => state.playbackSpeed);
  const setIsPlaying = useSessionStore((state) => state.setIsPlaying);
  const setPlaybackSpeed = useSessionStore((state) => state.setPlaybackSpeed);
  const nextStep = useSessionStore((state) => state.nextStep);
  const previousStep = useSessionStore((state) => state.previousStep);
  const setCurrentStep = useSessionStore((state) => state.setCurrentStep);

  // Auto-play animation
  React.useEffect(() => {
    if (!isPlaying || !session) return;

    const interval = setInterval(() => {
      if (currentStep < session.processor_states.length - 1) {
        nextStep();
      } else {
        setIsPlaying(false);
      }
    }, playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, currentStep, session, playbackSpeed, nextStep, setIsPlaying]);

  if (!session) return null;

  const progress = (currentStep / (session.processor_states.length - 1)) * 100;

  return (
    <div className="flex items-center gap-4 p-4 bg-white border-t">
      {/* Play/Pause */}
      <button
        onClick={() => setIsPlaying(!isPlaying)}
        className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600"
      >
        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
      </button>

      {/* Previous/Next */}
      <button
        onClick={previousStep}
        disabled={currentStep === 0}
        className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
      >
        <SkipBack size={20} />
      </button>
      <button
        onClick={nextStep}
        disabled={currentStep === session.processor_states.length - 1}
        className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
      >
        <SkipForward size={20} />
      </button>

      {/* Timeline */}
      <div className="flex-1">
        <input
          type="range"
          min="0"
          max={session.processor_states.length - 1}
          value={currentStep}
          onChange={(e) => setCurrentStep(Number(e.target.value))}
          className="w-full"
        />
        <div className="text-xs text-gray-600 mt-1">
          Step {currentStep} / {session.processor_states.length - 1}
        </div>
      </div>

      {/* Speed Control */}
      <select
        value={playbackSpeed}
        onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
        className="px-3 py-2 border rounded text-sm"
      >
        <option value={2000}>0.5x</option>
        <option value={1000}>1x</option>
        <option value={500}>2x</option>
        <option value={250}>4x</option>
      </select>
    </div>
  );
};
```

### Phase 7: Main App (20 minutes)

Create `src/App.tsx`:

```typescript
import React from 'react';
import { WorkflowGraph } from './components/workflow/WorkflowGraph';
import { ImageViewer } from './components/viewer/ImageViewer';
import { PlaybackControls } from './components/playback/PlaybackControls';
import { useSessionStore } from './stores/sessionStore';

function App() {
  const loadSession = useSessionStore((state) => state.loadSession);

  // Load session from JSON file
  React.useEffect(() => {
    // In production, you'd load this from a URL parameter or file upload
    fetch('/sample-session.json')
      .then((res) => res.json())
      .then((session) => loadSession(session))
      .catch((err) => console.error('Failed to load session:', err));
  }, [loadSession]);

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 text-white p-4">
        <h1 className="text-xl font-bold">OMR Workflow Visualization</h1>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph Panel */}
        <div className="w-1/3 border-r">
          <WorkflowGraph />
        </div>

        {/* Image Viewer */}
        <div className="flex-1">
          <ImageViewer />
        </div>
      </div>

      {/* Playback Controls */}
      <PlaybackControls />
    </div>
  );
}

export default App;
```

## Migration Checklist

- [ ] Phase 1: Project setup with Vite + TypeScript
- [ ] Phase 2: Define TypeScript types
- [ ] Phase 3: Setup Zustand store
- [ ] Phase 4: Create WorkflowGraph component
- [ ] Phase 5: Create ImageViewer component
- [ ] Phase 6: Create PlaybackControls component
- [ ] Phase 7: Integrate all components in App
- [ ] Phase 8: Add unit tests
- [ ] Phase 9: Add E2E tests with Playwright
- [ ] Phase 10: Add file upload functionality
- [ ] Phase 11: Add export functionality
- [ ] Phase 12: Optimize bundle size
- [ ] Phase 13: Deploy to production

## Next Steps

1. **Try it locally:**
   ```bash
   cd workflow-viz-app
   npm run dev
   ```

2. **Add features:**
   - File upload for JSON sessions
   - Export to video/GIF
   - Comparison mode (2 sessions side-by-side)
   - Keyboard shortcuts
   - Dark mode
   - Search/filter processors

3. **Deploy:**
   ```bash
   npm run build
   # Deploy to Vercel/Netlify/GitHub Pages
   ```

## Estimated Migration Time

- **Minimal viable version**: 4-6 hours
- **Full-featured version**: 2-3 days
- **With tests + polish**: 1 week

## Resources

- [React Flow Documentation](https://reactflow.dev/)
- [Zustand Documentation](https://github.com/pmndrs/zustand)
- [Tailwind CSS](https://tailwindcss.com/)
- [Vite Guide](https://vitejs.dev/guide/)

