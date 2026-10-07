# Build context is the repo root so the browser can reuse node/bench.js
FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY web/public/ /usr/share/nginx/html/
COPY node/bench.js /usr/share/nginx/html/bench.js
