import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { hash, otp6 } from "@/lib/crypto";
import { sendCode } from "@/lib/mail";
import { z } from "zod";

const schema = z.object({
  email: z.string().email(),
  password: z.string().max(12).regex(/^[A-Za-z0-9]+$/),
});

export async function POST(req: Request) {
  const body = await req.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: "Invalid" }, { status: 400 });

  const { email, password } = parsed.data;
  const passwordHash = await hash(password);

  await prisma.user.upsert({
    where: { email },
    update: { passwordHash },
    create: { email, passwordHash },
  });

  // sinh OTP và LÀM SẠCH token cũ -> tạo mới (tránh lệch truy vấn)
  const code = otp6();
  const tokenHash = await hash(code);
  const expiresAt = new Date(Date.now() + 10 * 60 * 1000);

  await prisma.token.deleteMany({ where: { email, kind: 'VERIFY' } });
  await prisma.token.create({
    data: { email, kind: 'VERIFY', tokenHash, expiresAt },
  });

  // (dev có thể bật log) console.log("REGISTER OTP", email, code);
  await sendCode(email, "FitAI – Verify your email", code);
  return NextResponse.json({ ok: true });
}
