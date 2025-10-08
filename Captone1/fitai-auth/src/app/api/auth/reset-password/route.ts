import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { hash, verify as verifyHash } from "@/lib/crypto";
import { z } from "zod";

const schema = z.object({
  email: z.string().email(),
  newPassword: z.string().max(12).regex(/^[A-Za-z0-9]+$/),
  resetSession: z.string(),
});

export async function POST(req: Request) {
  const body = await req.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: "Invalid" }, { status: 400 });

  const { email, newPassword, resetSession } = parsed.data;

  const token = await prisma.token.findFirst({
    where: { email, kind: 'RESET' },
  });

  if (!token || token.expiresAt < new Date())
    return NextResponse.json({ error: "Session expired" }, { status: 400 });

  const ok = await verifyHash(resetSession, token.tokenHash);
  if (!ok) return NextResponse.json({ error: "Invalid session" }, { status: 400 });

  await prisma.$transaction([
    prisma.user.update({
      where: { email },
      data: { passwordHash: await hash(newPassword) },
    }),
    prisma.token.deleteMany({ where: { email, kind: 'RESET' } }),
  ]);

  return NextResponse.json({ ok: true });
}
