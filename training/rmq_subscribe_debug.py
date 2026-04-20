import argparse
import json
import pika
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="206.180.209.81")
    ap.add_argument("--port", type=int, default=5672)
    ap.add_argument("--user", default="twin")
    ap.add_argument("--password", default="twinrabbitpass")
    ap.add_argument("--vhost", default="/")
    ap.add_argument("--exchange", default="cv.events")
    ap.add_argument("--binding", default="cv.zone.*", help="topic binding key")
    ap.add_argument("--pretty", action="store_true", help="pretty print json")
    args = ap.parse_args()

    creds = pika.PlainCredentials(args.user, args.password)
    params = pika.ConnectionParameters(
        host=args.host,
        port=args.port,
        virtual_host=args.vhost,
        credentials=creds,
        heartbeat=30,
        blocked_connection_timeout=30,
    )

    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    # Ensure exchange exists (same settings as publisher)
    ch.exchange_declare(exchange=args.exchange, exchange_type="topic", durable=True)

    # Create a temporary exclusive queue that auto-deletes when this script exits
    q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    queue_name = q.method.queue

    ch.queue_bind(queue=queue_name, exchange=args.exchange, routing_key=args.binding)

    print(f"[{datetime.utcnow().isoformat()}] Listening on exchange='{args.exchange}' binding='{args.binding}'")
    print(f"Temporary queue: {queue_name}")
    print("Press Ctrl+C to stop.\n")

    def on_msg(ch, method, properties, body):
        ts = datetime.utcnow().isoformat()
        print(f"\n[{ts}] rk={method.routing_key} bytes={len(body)}")

        try:
            payload = json.loads(body.decode("utf-8"))
            if args.pretty:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(payload)
        except Exception:
            print(body[:500])

        ch.basic_ack(method.delivery_tag)

    ch.basic_qos(prefetch_count=50)
    ch.basic_consume(queue=queue_name, on_message_callback=on_msg, auto_ack=False)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            ch.stop_consuming()
        except Exception:
            pass
        conn.close()

if __name__ == "__main__":
    main()