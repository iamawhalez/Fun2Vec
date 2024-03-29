# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import capstone
import binascii
from capstone import *


def filter_memory_references(i):
    inst = "" + i.mnemonic
    for op in i.operands:
        if op.type == 1:
            inst = inst + " " + i.reg_name(op.reg)
        elif op.type == 2:
            imm = int(op.imm)
            if -int(5000) <= imm <= int(5000):
                inst = inst + " " + str(hex(op.imm))
            else:
                inst = inst + " " + str("HIMM")
        elif op.type == 3:
            mem = op.mem
            if mem.base == 0:
                r = "[" + "MEM" + "]"
            else:
                r = (
                    "["
                    + str(i.reg_name(mem.base))
                    + "*"
                    + str(mem.scale)
                    + "+"
                    + str(mem.disp)
                    + "]"
                )
                inst = inst + " " + r
        if len(i.operands) > 1:
            inst = inst + ","
    if "," in inst:
        inst = inst[:-1]
    inst = inst.replace(" ", "_")
    return str(inst)


def disassemble(asm, arch, bits, verbose=False):
    binary = binascii.unhexlify(asm)

    if arch == "x86":
        cs_arch = capstone.CS_ARCH_X86
    elif arch == "x64":
        cs_arch = capstone.CS_ARCH_X86
        arch = 'x86'
    elif arch == "arm":
        cs_arch = capstone.CS_ARCH_ARM
    else:
        if verbose:
            print("Architecture not supported")
        return

    if bits == 32:
        cs_bits = capstone.CS_MODE_32
    elif bits == 64:
        cs_bits = capstone.CS_MODE_64
    else:
        cs_bits = capstone.CS_MODE_64
    mds = {}
    mds['x86-32'] = Cs(CS_ARCH_X86, CS_MODE_32)
    mds['x86-64'] = Cs(CS_ARCH_X86, CS_MODE_64)
    mds['arm-32'] = Cs(CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN)
    mds['arm-64'] = Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN)

    md = mds[f'{arch}-{bits}']
    # md = capstone.Cs(cs_arch, cs_bits)
    md.detail = True
    instructions = []

    if arch == "x86":
        for i in md.disasm(binary, 0):
            instructions.append("X_" + filter_memory_references(i))
    elif arch == "arm":     
        for i in md.disasm(binary, 0):
            instructions.append("A_" + filter_memory_references(i))
    return instructions
