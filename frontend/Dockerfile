# ── Stage 1: Node 22 builder ──────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package.json only — no lockfile required, npm install generates it
COPY package.json ./

# Install deps (generates package-lock.json inside the image)
RUN npm install --silent

# Copy source and build
COPY . .
RUN npm run build


# ── Stage 2: nginx static server ─────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove default nginx site
RUN rm /etc/nginx/conf.d/default.conf

# Copy our nginx config (SPA routing + /api proxy + SSE support)
COPY nginx.conf /etc/nginx/conf.d/app.conf

# Copy Vite build output
COPY --from=builder /app/dist /usr/share/nginx/html

# Allow nginx to run as non-root
RUN chown -R nginx:nginx /usr/share/nginx/html \
    && chown -R nginx:nginx /var/cache/nginx \
    && chown -R nginx:nginx /var/log/nginx \
    && touch /var/run/nginx.pid \
    && chown -R nginx:nginx /var/run/nginx.pid

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
