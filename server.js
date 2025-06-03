// server.js
import { WebSocketServer } from 'ws';
const PORT = process.env.PORT || 3000;

const wss = new WebSocketServer({ port: PORT }, () => {
  console.log(`✅ WebSocket server running on port ${PORT}`);
});

wss.on('connection', (ws) => {
  console.log('🔌 Client connected');

  ws.on('message', (message) => {
    console.log('📨 Received:', message);
    for (const client of wss.clients) {
      if (client !== ws && client.readyState === ws.OPEN) {
        client.send(message);
      }
    }
  });

  ws.on('close', () => {
    console.log('❌ Client disconnected');
  });
});
