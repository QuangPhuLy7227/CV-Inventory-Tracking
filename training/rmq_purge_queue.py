import argparse
import pika

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="206.180.209.81")
    ap.add_argument("--port", type=int, default=5672)
    ap.add_argument("--user", default="twin")
    ap.add_argument("--password", default="twinrabbitpass")
    ap.add_argument("--vhost", default="/")

    ap.add_argument("--queue", required=True, help="Queue name to purge/delete, e.g. backend.cv.events")
    ap.add_argument("--delete", action="store_true", help="Delete queue instead of purging")
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

    try:
        if args.delete:
            ch.queue_delete(queue=args.queue)
            print(f"✅ Deleted queue: {args.queue}")
        else:
            res = ch.queue_purge(queue=args.queue)
            # res.method.message_count is how many were purged
            print(f"✅ Purged queue: {args.queue} (removed {res.method.message_count} messages)")
    finally:
        conn.close()

if __name__ == "__main__":
    main()