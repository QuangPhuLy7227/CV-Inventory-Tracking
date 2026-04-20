import json
import pika
from typing import Any, Dict, Optional

class RabbitMQPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        exchange: str = "cv.events",
        exchange_type: str = "topic",
        vhost: str = "/",
        enable_confirm: bool = False,
    ):
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.vhost = vhost
        self.exchange = exchange
        self.exchange_type = exchange_type
        self.enable_confirm = enable_confirm

        self.conn: Optional[pika.BlockingConnection] = None
        self.ch: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self._connect()

    def _connect(self):
        creds = pika.PlainCredentials(self.username, self.password)
        params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=creds,
            heartbeat=30,
            blocked_connection_timeout=30,
            connection_attempts=3,
            retry_delay=2,
        )
        self.conn = pika.BlockingConnection(params)
        self.ch = self.conn.channel()
        self.ch.exchange_declare(exchange=self.exchange, exchange_type=self.exchange_type, durable=True)

        if self.enable_confirm:
            self.ch.confirm_delivery()

    def publish(self, routing_key: str, message: Dict[str, Any]) -> None:
        if not self.conn or self.conn.is_closed or not self.ch or self.ch.is_closed:
            self._connect()

        body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        try:
            ok = self.ch.basic_publish(
                exchange=self.exchange,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                ),
            )
            # If confirm_delivery is enabled, ok can be False if broker didn't confirm.
            if self.enable_confirm and not ok:
                raise RuntimeError("RabbitMQ publish not confirmed")
        except Exception:
            # one reconnect attempt then retry once
            self._safe_close()
            self._connect()
            ok = self.ch.basic_publish(
                exchange=self.exchange,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                ),
            )
            if self.enable_confirm and not ok:
                raise RuntimeError("RabbitMQ publish not confirmed after reconnect")

    def _safe_close(self):
        try:
            if self.ch and not self.ch.is_closed:
                self.ch.close()
        except Exception:
            pass
        try:
            if self.conn and not self.conn.is_closed:
                self.conn.close()
        except Exception:
            pass

    def close(self):
        self._safe_close()