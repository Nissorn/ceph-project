# ── Stage 1: Build Astro static site ──────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Install only production deps (no dev server, no source map tools)
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

COPY . .

# Build static output into /app/dist
RUN npm run build

# ── Stage 2: Static file server ───────────────────────────────────────────────
FROM node:22-alpine

WORKDIR /app

# security: non-root
RUN addgroup -g 1001 -S appgroup && \
    adduser  -u 1001 -S appuser -G appgroup

# Install serve or nginx-like static server
# Using npm serve as a lightweight option; replace with nginx for CDN-style if needed
RUN npm install -g serve

COPY --from=builder --chown=appuser:appgroup /app/dist ./dist

# VITE_API_URL is embedded at build time via --env flag; set here as build arg
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

USER appuser

EXPOSE 4321

# Health: serve responds on port 4321
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:4321/ || exit 1

CMD ["serve", "-s", "dist", "-l", "4321", "--no-clipboard"]