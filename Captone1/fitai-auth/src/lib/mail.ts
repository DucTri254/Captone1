import nodemailer from "nodemailer";

export async function sendCode(to: string, subject: string, code: string) {
  const { SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL } = process.env;
  if (!SMTP_HOST || !SMTP_USER || !SMTP_PASS) {
    console.log(`[DEV] Send mail to ${to}: ${subject} -> CODE: ${code}`);
    return;
  }
  const port = Number(SMTP_PORT || 587);
  const transporter = nodemailer.createTransport({
    host: SMTP_HOST,
    port,
    secure: port === 465, // 465=SSL, 587=STARTTLS
    auth: { user: SMTP_USER, pass: SMTP_PASS },
  });
  await transporter.sendMail({
    from: FROM_EMAIL || SMTP_USER,
    to,
    subject,
    text: `Your code is: ${code} (valid 10 minutes)`,
    html: `<p>Your code is:</p>
           <p style="font-size:24px;font-weight:700;letter-spacing:4px">${code}</p>
           <p>Valid 10 minutes.</p>`,
  });
}
