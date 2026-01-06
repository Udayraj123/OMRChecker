# OMR Workflow Visualization - React App

Modern React-based visualization tool for OMR processing workflows.

## Features

- 🎯 **Interactive Workflow Graph** - Powered by React Flow
- 🖼️ **Image Viewer** - View processor outputs with zoom controls
- ▶️ **Playback Controls** - Animate through workflow steps
- 📊 **Metadata Display** - See processor timing and details
- 🎨 **Modern UI** - Built with Tailwind CSS
- ⚡ **Fast** - Vite for lightning-fast development
- 🔒 **Type Safe** - Full TypeScript support
- ✅ **Tested** - Vitest + React Testing Library

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run tests
npm test
```

## Usage

### Load a Session

1. **From URL Parameter:**
   ```
   http://localhost:3000?session=path/to/session.json
   ```

2. **From File Upload:**
   - Click "Load Session" button
   - Select JSON file from your computer

3. **From Sample:**
   - App loads sample session by default

### Keyboard Shortcuts

- `Space` - Play/Pause
- `←` - Previous step
- `→` - Next step
- `Home` - First step
- `End` - Last step
- `1-4` - Set playback speed

### Controls

- **Graph Panel** - Click nodes to jump to that processor
- **Image Viewer** - Zoom in/out, toggle grayscale/colored
- **Timeline** - Scrub to any point in the workflow
- **Speed Control** - Adjust animation speed (0.5x - 4x)

## Project Structure

```
src/
├── components/     # React components
├── hooks/          # Custom React hooks
├── stores/         # Zustand state stores
├── types/          # TypeScript types
├── utils/          # Utility functions
└── App.tsx         # Main app component
```

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **React Flow** - Workflow visualization
- **Zustand** - State management
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **Framer Motion** - Animations
- **Vitest** - Testing

## Development

### Adding New Features

1. Create component in `src/components/`
2. Add types to `src/types/`
3. Update store in `src/stores/` if needed
4. Write tests in `*.test.tsx`

### Code Quality

```bash
# Type check
npm run type-check

# Lint
npm run lint

# Format (if using Prettier)
npm run format
```

## Deployment

### Build

```bash
npm run build
```

Output will be in `dist/` directory.

### Deploy to Vercel

```bash
npm i -g vercel
vercel
```

### Deploy to Netlify

```bash
npm i -g netlify-cli
netlify deploy --prod --dir=dist
```

### Deploy to GitHub Pages

```bash
npm run build
# Push dist/ folder to gh-pages branch
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - Same as OMRChecker main project

