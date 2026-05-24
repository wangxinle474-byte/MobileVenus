"""Read a UTF-16 (PowerShell redirect) or UTF-8 log file and print last N lines."""
import sys


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    filt = sys.argv[3] if len(sys.argv) > 3 else None

    with open(path, 'rb') as f:
        data = f.read()
    if data[:2] == b'\xff\xfe':
        txt = data.decode('utf-16-le', errors='ignore')
    elif data[:2] == b'\xfe\xff':
        txt = data.decode('utf-16-be', errors='ignore')
    else:
        txt = data.decode('utf-8', errors='ignore')

    lines = [l for l in txt.split('\n') if l.strip()]
    if filt:
        lines = [l for l in lines if any(k in l for k in filt.split('|'))]
    for l in lines[-n:]:
        print(l)


if __name__ == '__main__':
    main()
