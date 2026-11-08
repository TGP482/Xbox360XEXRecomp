# Flat PE of a XEX for Ghidra: the loaded image at its base, recompiler functions and config names as exports.
# usage, from the game repo after a build:
#   xex_symbols --image assets/default.xex --out image.bin
#   python sdk/tools/symbols/xex2pe.py config/game.toml image.bin game.exe
# Ghidra imports it as a PE; pick PowerPC big endian 64 bit with 32 bit addresses as the language.
import re
import struct
import sys
import tomllib
from pathlib import Path


def load(path, cfg):
    t = tomllib.loads(path.read_text())
    for inc in t.get('includes', []):
        load(path.parent / inc, cfg)
    for k in ('xex', 'functions', 'symbols'):
        cfg.setdefault(k, {}).update(t.get(k, {}))
    return cfg


cfg_path, image_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cfg = load(cfg_path, {})
root = cfg_path.parent.parent
raw = image_path.read_bytes()
base, entry = struct.unpack_from('<II', raw)
img = bytearray(raw[8:])
isize = (len(img) + 0xFFF) & ~0xFFF
img += bytes(isize - len(img))

names = {}
mapping = root / cfg['xex']['out'] / 'ppc_func_mapping.cpp'
if mapping.exists():
    for a in re.findall(r'\{0x([0-9A-F]+)u, sub_', mapping.read_text()):
        names[int(a, 16)] = 'sub_%08X' % int(a, 16)
for va, name in cfg['functions'].items():
    names[int(va, 0)] = name.replace('::', '_')
for name, va in cfg['symbols'].items():
    if va:
        names[va] = name

exp = sorted(names)
nm = [names[a].encode() for a in exp]
order = sorted(range(len(exp)), key=lambda i: nm[i])
# export directory (40 bytes), address table, name pointers, ordinals and names, page aligned after the image
ert = isize
eat = ert + 40
npt = eat + 4 * len(exp)
ot = npt + 4 * len(exp)
st = ot + 2 * len(exp)
strs, off = bytearray(b'image.exe\0'), {}
for i in range(len(exp)):
    off[i] = st + len(strs)
    strs += nm[i] + b'\0'
edata = struct.pack('<IIHHIIIIIII', 0, 0, 0, 0, st, 1, len(exp), len(exp), eat, npt, ot)
edata += b''.join(struct.pack('<I', a - base) for a in exp)
edata += b''.join(struct.pack('<I', off[i]) for i in order)
edata += b''.join(struct.pack('<H', i) for i in order)
edata += strs
esz = (len(edata) + 0xFFF) & ~0xFFF

# PE32 optional header, then the export directory entry and 15 empty ones
opt = struct.pack('<HBBIIIIIIIIIHHHHHHIIIIHHIIIIII', 0x10B, 14, 0, isize, 0, 0, entry - base, 0x1000, 0x1000, base,
                  0x1000, 0x1000, 4, 0, 0, 0, 4, 0, 0, ert + esz, 0x1000, 0, 2, 0, 0x100000, 0x1000, 0x100000, 0x1000,
                  0, 16)
opt += struct.pack('<II', ert, len(edata)) + bytes(8 * 15)
coff = struct.pack('<HHIIIHH', 0x1F2, 2, 0, 0, 0, len(opt), 0x0102)  # PowerPC big endian (Xbox 360)
flat = struct.pack('<8sIIIIIIHHI', b'.flat', isize - 0x1000, 0x1000, isize - 0x1000, 0x1000, 0, 0, 0, 0, 0xE0000060)
exps = struct.pack('<8sIIIIIIHHI', b'.edata', len(edata), ert, esz, ert, 0, 0, 0, 0, 0x40000040)
dos = bytearray(0x40)
dos[:2] = b'MZ'
struct.pack_into('<I', dos, 0x3C, 0x40)
head = bytes(dos) + b'PE\0\0' + coff + opt + flat + exps

f = bytearray(ert + esz)
f[0x1000:isize] = img[0x1000:]
f[:len(head)] = head
f[ert:ert + len(edata)] = edata
out.write_bytes(f)
print(out, len(exp), 'exports')
