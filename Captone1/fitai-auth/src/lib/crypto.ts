import bcrypt from "bcryptjs";

export const hash = (s: string) => bcrypt.hash(s, 10);
export const verify = (s: string, h: string) => bcrypt.compare(s, h);
export const otp6 = () =>
  String(Math.floor(100000 + Math.random() * 900000)); // 6 digits
