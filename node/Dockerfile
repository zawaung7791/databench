FROM node:22-alpine
WORKDIR /app
COPY package.json bench.js worker.js server.js ./
USER node
EXPOSE 8000
CMD ["node", "server.js"]
