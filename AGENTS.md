# Repository Guidelines

## Project Structure & Module Organization
This repository is a TypeScript full-stack app with a Vite React client and a small Node server.

- `client/`: frontend app (`index.html`, `src/`, static files in `public/`)
- `server/`: backend entry (`index.ts`) bundled for Node runtime
- `shared/`: shared constants/types used across client and server
- `patches/`: `pnpm` patch files (currently `wouter@3.7.1.patch`)
- `dist/`: build output (`dist/public` for frontend assets, `dist/index.js` for server)

Use path aliases from `tsconfig.json`: `@/*` -> `client/src/*`, `@shared/*` -> `shared/*`.

## Build, Test, and Development Commands
- `pnpm install`: install dependencies (required first step)
- `pnpm dev`: start local development server via Vite (`--host`, default port around `3000`)
- `pnpm build`: build frontend and bundle backend into `dist/`
- `pnpm start`: run production build (`NODE_ENV=production node dist/index.js`)
- `pnpm preview`: preview built frontend locally
- `pnpm check`: run TypeScript type checking (`tsc --noEmit`)
- `pnpm format`: format repository with Prettier

## Coding Style & Naming Conventions
- Language: TypeScript (`strict` mode enabled)
- Formatting: Prettier (`tabWidth: 2`, `printWidth: 80`, `semi: true`, `endOfLine: lf`)
- Indentation: 2 spaces, no tabs
- React components: `PascalCase` filenames (e.g., `Map.tsx`, `ErrorBoundary.tsx`)
- Hooks/utilities: `camelCase` filenames/functions (e.g., `useMobile.tsx`, `utils.ts`)
- Prefer alias imports (`@/...`, `@shared/...`) over deep relative paths when practical.

## Testing Guidelines
No dedicated automated test suite is currently committed. Before opening a PR:
- run `pnpm check`
- run `pnpm build`
- manually verify key flows in `pnpm dev` (navigation, map, dialogs, theme behavior)

When adding tests, use Vitest (already in dev dependencies) and place files as `*.test.ts` or `*.test.tsx` near related modules.

## Commit & Pull Request Guidelines
Git history is not available in this workspace snapshot, so follow Conventional Commits:
- `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`, `chore: ...`

PRs should include:
- a short problem/solution summary
- linked issue/ticket (if available)
- screenshots or short recordings for UI changes
- explicit notes for config, dependency, or patch updates (especially under `patches/`)
