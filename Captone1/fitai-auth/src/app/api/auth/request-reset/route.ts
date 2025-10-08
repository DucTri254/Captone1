import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { hash, otp6 } from "@/lib/crypto";
import { sendCode } from "@/lib/mail";

export async function POST(req: Request) {
  const { email } = await req.json();

  const code = otp6();
  const tokenHash = await hash(code);
  const expiresAt = new Date(Date.now() + 10 * 60 * 1000);

  await prisma.token.deleteMany({ where: { email, kind: 'RESET' } });
  await prisma.token.create({
    data: { email, kind: 'RESET', tokenHash, expiresAt },
  });

  // (dev) console.log("RESET OTP", email, code);
  await sendCode(email, "FitAI – Password reset code", code);
  return NextResponse.json({ ok: true });
}
