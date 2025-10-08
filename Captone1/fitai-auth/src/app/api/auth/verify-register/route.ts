import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { verify as verifyHash } from "@/lib/crypto";

export async function POST(req: Request) {
  const { email, code } = await req.json();
  const normalized = String(code || "").replace(/\s/g, "").trim();

  // tìm token bằng findFirst thay vì findUnique (không phụ thuộc khóa tổng hợp)
  const token = await prisma.token.findFirst({
    where: { email, kind: 'VERIFY' },
  });

  if (!token || token.usedAt || token.expiresAt < new Date())
    return NextResponse.json({ ok: false, reason: "expired" }, { status: 400 });

  const ok = await verifyHash(normalized, token.tokenHash);
  if (!ok) return NextResponse.json({ ok: false, reason: "mismatch" }, { status: 400 });

  await prisma.$transaction([
    prisma.user.update({ where: { email }, data: { emailVerifiedAt: new Date() } }),
    prisma.token.deleteMany({ where: { email, kind: 'VERIFY' } }),
  ]);

  return NextResponse.json({ ok: true });
}
