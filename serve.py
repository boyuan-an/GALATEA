#!/usr/bin/env python3
"""Preview this static website locally, including byte-range video seeking."""

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.webp': 'image/webp',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def send_head(self):
        self.remaining = None
        path = self.translate_path(self.path)
        value = self.headers.get('Range', '')
        if not value or not os.path.isfile(path):
            return super().send_head()
        # Serve a single byte range; ignore unsupported multi-range requests.
        if not value.startswith('bytes=') or ',' in value:
            return super().send_head()
        stream = open(path, 'rb')
        size = os.fstat(stream.fileno()).st_size
        try:
            first, last = value[6:].split('-', 1)
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                suffix = int(last)
                if suffix <= 0:
                    raise ValueError
                start, end = max(0, size - suffix), size - 1
            if start < 0 or start >= size or end < start:
                raise ValueError
        except ValueError:
            stream.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        stream.seek(start)
        self.remaining = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining > 0:
                data = source.read(min(self.remaining, 64 * 1024))
                if not data:
                    break
                outputfile.write(data)
                self.remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel video requests when seeking or navigating.


def main():
    parser = argparse.ArgumentParser(
        description='GALATEA local website preview server (Python standard library only)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='bind address (defaults to all interfaces for local-network access)',
    )
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        print(f'Listening on {args.host}:{server.server_port}', flush=True)
        print(
            'Press Ctrl+C to stop the preview server. '
            'Refresh the browser after editing files.',
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
