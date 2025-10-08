import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { verify as verifyHash, hash } from "@/lib/crypto";

export async function POST(req: Request) {
  const { email, code } = await req.json();
  const normalized = String(code || "").replace(/\s/g, "").trim();

  const token = await prisma.token.findFirst({
    where: { email, kind: 'RESET' },
  });

  if (!token || token.usedAt || token.expiresAt < new Date())
    return NextResponse.json({ ok: false, reason: "expired" }, { status: 400 });

  const ok = await verifyHash(normalized, token.tokenHash);
  if (!ok) return NextResponse.json({ ok: false, reason: "mismatch" }, { status: 400 });

  const resetSession = crypto.randomUUID();

  await prisma.token.updateMany({
    where: { email, kind: 'RESET' },
    data: {
      tokenHash: await hash(resetSession),
      expiresAt: new Date(Date.now() + 5 * 60 * 1000),
      usedAt: null,
    },
  });

  return NextResponse.json({ ok: true, resetSession });
}
