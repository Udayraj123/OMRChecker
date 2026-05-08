# MinSU AI OMR Scanner Web UI

Next.js dashboard for the OMRChecker project. It provides a login-only admin interface for uploading OMR sheets, viewing scan results, exporting XLS files, editing answer keys, and tracking live upload history.

## Live App

Production deployment:

```text
https://web-ui-zeta-tawny.vercel.app
```

Default login:

```text
Username: kenji
Password: 12345
```

## Features

- Login-only access with a signed HTTP-only session cookie.
- Protected dashboard, upload, results, analytics, students, answer-key, and history pages.
- PDF, JPG, JPEG, and PNG upload support.
- Vercel upload-size guard for files over 4 MB.
- Local Python OMRChecker processing for development.
- MySQL-backed scan storage when a database is available.
- Browser local-storage fallback when MySQL is offline.
- XLS export for selected rows or all scan rows.
- Duplicate detection by application number, LRN, full name, and source file.
- Live upload history with automatic refresh and upload status tracking.

## Local Setup

Install dependencies:

```powershell
cd C:\Users\nameyourpc\Downloads\OMRChecker-codex-minsu-web-ui\OMRChecker-codex-minsu-web-ui\web-ui
npm.cmd install
```

Create a new file named `.env.local` inside `web-ui`, or copy `web-ui\.env.local.example` to `web-ui\.env.local`.

Edit the paths and replace `nameyourpc` with your Windows user folder name. This prevents the `OMR_REPO_ROOT is not configured` error and makes the scanner use the Python version with OpenCV installed:

```env
OMR_REPO_ROOT=C:\Users\nameyourpc\Downloads\OMRChecker-codex-minsu-web-ui\OMRChecker-codex-minsu-web-ui
OMR_PYTHON=C:\Users\nameyourpc\AppData\Local\Programs\Python\Python310\python.exe
```

Run the app:

```powershell
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:3000/dashboard
```

If you are not logged in, the app redirects to `/login`.

## Database Import

The importable MySQL schema is available at:

```text
database/schema.sql
public/database/minsu_omr_scanner_import.sql
```

Import either file in Laragon/phpMyAdmin or through MySQL to create the `minsu_omr_scanner` database and `scan_results` table.

## Upload Notes

The deployed Vercel app cannot access MySQL or Python running on a local PC. On Vercel, the app gracefully falls back to browser-local records when the database is unavailable. For shared real-time records across devices, configure a hosted database such as Vercel Postgres, Neon, Supabase, or hosted MySQL.

For local scanning, the upload API calls the Python OMRChecker backend through `OMR_PYTHON` and `OMR_REPO_ROOT`.

The scan route also auto-detects the parent OMRChecker folder when possible. The current `web-ui/src/app/api/scan/route.ts` should keep this fallback logic:

```ts
function resolveOmrRepoRoot() {
  const configuredRoot = process.env.OMR_REPO_ROOT?.trim();
  if (configuredRoot) return configuredRoot;

  const bundledRoot = path.join(/* turbopackIgnore: true */ process.cwd(), "..");
  const hasBundledOmrFiles =
    existsSync(path.join(bundledRoot, "main.py")) &&
    existsSync(path.join(bundledRoot, "inputs", "template.json")) &&
    existsSync(path.join(bundledRoot, "inputs", "image.jpg"));

  return hasBundledOmrFiles ? bundledRoot : "";
}
```

## Useful Commands

```bash
npm run lint
npm run build
vercel deploy --prod --yes
```

## Project Paths

```text
src/app/login/page.tsx          Login page
src/lib/auth.ts                 Hardcoded auth and session helpers
src/proxy.ts                    Route protection
src/app/api/scan/route.ts       OMR upload processing API
src/app/api/results/route.ts    Scan results API
src/components/dashboard-pages.tsx
src/components/scan-store.ts
database/schema.sql
```
