#!/usr/bin/env python3
import subprocess
import re
import argparse

def get_serial_mappings():
    """Get mapping of all serial devices to their serial numbers"""
    #
    # NOTE (macOS): Serial ports (IOCalloutDevice/IODialinDevice) live under
    # IOSerialBSDClient, not the IOUSB plane. Using -t gives us a tree that
    # includes USB product/vendor/serial info higher up the chain.
    #
    cmd = ["ioreg", "-r", "-c", "IOSerialBSDClient", "-l", "-t", "-w0"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to run ioreg. "
            f"exit={result.returncode}, stderr={result.stderr.strip()}"
        )

    mappings: dict[str, dict] = {}

    # Track the nearest ancestor USB metadata by indentation.
    # Stack items are (indent_index_of_first_quote, context_dict).
    stack: list[tuple[int, dict]] = []

    def _extract_string(line: str, key: str) -> str | None:
        m = re.search(rf'"{re.escape(key)}"\s*=\s*"([^"]*)"', line)
        return m.group(1) if m else None

    def _extract_int(line: str, key: str) -> int | None:
        m = re.search(rf'"{re.escape(key)}"\s*=\s*(\d+)', line)
        return int(m.group(1)) if m else None

    for raw in result.stdout.splitlines():
        if '"' not in raw:
            continue
        indent = raw.find('"')

        # Pop contexts until the top is at most as indented as this line.
        # If the indent is the same, we treat it as the same node continuing
        # (ioreg prints multiple properties of the same node at the same indent).
        while stack and stack[-1][0] > indent:
            stack.pop()

        if stack and stack[-1][0] == indent:
            ctx = stack[-1][1]
        else:
            parent_ctx = stack[-1][1] if stack else {}
            ctx = dict(parent_ctx)
            stack.append((indent, ctx))

        usb_serial = _extract_string(raw, "USB Serial Number")
        if usb_serial is not None:
            ctx["serial"] = usb_serial

        product = _extract_string(raw, "USB Product Name")
        if product is not None:
            ctx["product"] = product

        vendor = _extract_string(raw, "USB Vendor Name")
        if vendor is not None:
            ctx["vendor"] = vendor

        id_vendor = _extract_int(raw, "idVendor")
        if id_vendor is not None:
            ctx["idVendor"] = id_vendor

        id_product = _extract_int(raw, "idProduct")
        if id_product is not None:
            ctx["idProduct"] = id_product

        callout = _extract_string(raw, "IOCalloutDevice")
        if callout is not None:
            mappings[callout] = {
                "serial": ctx.get("serial", "Unknown"),
                "product": ctx.get("product", "Unknown"),
                "vendor": ctx.get("vendor", "Unknown"),
                "idVendor": ctx.get("idVendor", "Unknown"),
                "idProduct": ctx.get("idProduct", "Unknown"),
            }

        dialin = _extract_string(raw, "IODialinDevice")
        if dialin is not None:
            mappings[dialin] = {
                "serial": ctx.get("serial", "Unknown"),
                "product": ctx.get("product", "Unknown"),
                "vendor": ctx.get("vendor", "Unknown"),
                "idVendor": ctx.get("idVendor", "Unknown"),
                "idProduct": ctx.get("idProduct", "Unknown"),
            }

    return mappings

# Get all mappings
mappings = get_serial_mappings()

# Print all devices
print("Serial Port Mappings:")
print("=" * 50)
for device, info in mappings.items():
    print(
        f"{device:30} -> Serial: {info['serial']:20} "
        f"Vendor: {info['vendor']} Product: {info['product']}"
    )

# Look up specific port
parser = argparse.ArgumentParser(description="Map macOS serial ports to USB metadata.")
parser.add_argument(
    "--port",
    default="/dev/tty.usbmodem11201",
    help="Serial port path to look up (e.g. /dev/tty.usbmodem11101)",
)
args = parser.parse_args()

specific_port = args.port
if specific_port in mappings:
    print(f"\n{specific_port} corresponds to:")
    print(f"  Serial: {mappings[specific_port]['serial']}")
    print(f"  Product: {mappings[specific_port]['product']}")
    print(f"  Vendor: {mappings[specific_port]['vendor']}")
else:
    print(f"\n{specific_port} not found in active devices")
    
    # Check if it exists as a file
    import os
    if os.path.exists(specific_port):
        print("But the port file exists - may be a Bluetooth or other non-USB device")