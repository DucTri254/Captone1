"use client";
import { AuthFlow } from "@/components/auth/AuthFlow";

export default function Home() {
  return (
    <main className="min-h-screen grid place-items-center p-4 bg-[radial-gradient(ellipse_at_center,_#0b0c0f_0%,_#050607_60%,_#000_100%)] text-white">
      <AuthFlow />
    </main>
  );
}
