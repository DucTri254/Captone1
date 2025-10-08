"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Eye, EyeOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

type Screen = "login" | "signup" | "otp" | "forgot" | "reset"
type OtpPurpose = "register" | "reset" | null

const cardVariants = {
  initial: { opacity: 0, y: 20, scale: 0.98 },
  animate: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.25 } },
  exit:    { opacity: 0, y: -20, scale: 0.98, transition: { duration: 0.2 } },
}

export function AuthFlow() {
  const [screen, setScreen] = React.useState<Screen>("login")
  const [loading, setLoading] = React.useState(false)
  const [showPass, setShowPass] = React.useState(false)
  const [email, setEmail] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [confirm, setConfirm] = React.useState("")
  const [code, setCode] = React.useState("")
  const [otpPurpose, setOtpPurpose] = React.useState<OtpPurpose>(null)
  const [resetSession, setResetSession] = React.useState<string>("")

  const isEmail = (v: string) => /\S+@\S+\.\S+/.test(v)
  const passOk = (v: string) => v.length >= 1 && v.length <= 12 && /^[A-Za-z0-9]+$/.test(v)

  async function onLogin() {
    alert("Login (mock)")
  }

  // REGISTER -> OTP
  async function onSignup() {
    if (!isEmail(email)) return alert("Email không hợp lệ")
    if (!passOk(password)) return alert("Mật khẩu ≤12 và không ký tự đặc biệt")
    if (password !== confirm) return alert("Xác nhận mật khẩu chưa khớp")

    setLoading(true)
    const res = await fetch("/api/auth/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })
    setLoading(false)
    if (!res.ok) return alert("Không thể gửi OTP. Thử lại.")
    setOtpPurpose("register")
    setScreen("otp")
  }

  async function verifyRegisterOtp() {
    setLoading(true);
    const normalized = code.replace(/\s/g, "").trim();   // <— thêm dòng này
    const res = await fetch("/api/auth/verify-register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, code: normalized }), // <— dùng normalized
  });
  setLoading(false);
  if (!res.ok) return alert("Mã không đúng hoặc đã hết hạn");
  alert("Xác minh email thành công!");
  setScreen("login");
  }

  // FORGOT -> OTP -> RESET
  async function onSendResetCode() {
    if (!isEmail(email)) return alert("Email không hợp lệ")
    setLoading(true)
    const res = await fetch("/api/auth/request-reset", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    })
    setLoading(false)
    if (!res.ok) return alert("Không thể gửi mã")
    setOtpPurpose("reset")
    setScreen("otp")
  }

  async function verifyResetOtp() {
    setLoading(true);
  const normalized = code.replace(/\s/g, "").trim();   // <— thêm dòng này
  const res = await fetch("/api/auth/verify-reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, code: normalized }), // <— dùng normalized
  });
  setLoading(false);
  if (!res.ok) return alert("Mã không đúng hoặc đã hết hạn");
  const data = await res.json();
  setResetSession(data.resetSession);
  setScreen("reset");
  }

  async function onReset() {
    if (!passOk(password)) return alert("Mật khẩu ≤12 & không ký tự đặc biệt")
    if (password !== confirm) return alert("Xác nhận mật khẩu chưa khớp")

    setLoading(true)
    const res = await fetch("/api/auth/reset-password", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, newPassword: password, resetSession }),
    })
    setLoading(false)
    if (!res.ok) return alert("Đổi mật khẩu thất bại")
    alert("Đổi mật khẩu thành công!")
    setScreen("login")
  }

  async function resendOtp() {
    if (otpPurpose === "register") return onSignup()
    if (otpPurpose === "reset") return onSendResetCode()
  }

  return (
    <div className="w-full max-w-[420px]">
      <AnimatePresence mode="wait" initial={false}>
        {screen === "login" && (
          <motion.div key="login" variants={cardVariants} initial="initial" animate="animate" exit="exit">
            <AuthCard>
              <div className="space-y-5">
                <Field label="Email">
                  <Input className="input-like" value={email} onChange={e=>setEmail(e.target.value)} placeholder="Enter email" />
                </Field>
                <Field label="Password">
                  <div className="relative">
                    <Input className="input-like pr-10" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Enter password" type={showPass? "text" : "password"} />
                    <button type="button" onClick={()=>setShowPass(s=>!s)} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/80">
                      {showPass ? <EyeOff size={18}/> : <Eye size={18}/>}
                    </button>
                  </div>
                </Field>
                <div className="text-sm text-white/60 -mt-2">
                  <button className="hover:underline" onClick={()=>setScreen("forgot")}>Forgot password?</button>
                </div>
                <AccentButton onClick={onLogin} loading={loading}>Login</AccentButton>
                <Separator className="bg-white/5" />
                <p className="text-center text-sm text-white/70">
                  Don&apos;t have an account?{" "}
                  <button className="hover:underline text-white" onClick={()=>setScreen("signup")}>Sign up</button>
                </p>
              </div>
            </AuthCard>
          </motion.div>
        )}

        {screen === "signup" && (
          <motion.div key="signup" variants={cardVariants} initial="initial" animate="animate" exit="exit">
            <AuthCard>
              <div className="space-y-5">
                <Field label="Email"><Input className="input-like" value={email} onChange={e=>setEmail(e.target.value)} placeholder="Enter email" /></Field>
                <Field label="Password"><Input className="input-like" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Enter password" type="password" /></Field>
                <Field label="Confirm password"><Input className="input-like" value={confirm} onChange={e=>setConfirm(e.target.value)} placeholder="Re-enter password" type="password" /></Field>
                <Separator className="bg-white/5" />
                <AccentButton onClick={onSignup} loading={loading}>Sign up</AccentButton>
                <p className="text-center text-sm text-white/70">
                  Already have an account?{" "}
                  <button className="hover:underline text-white" onClick={()=>setScreen("login")}>Sign in</button>
                </p>
              </div>
            </AuthCard>
          </motion.div>
        )}

        {screen === "forgot" && (
          <motion.div key="forgot" variants={cardVariants} initial="initial" animate="animate" exit="exit">
            <AuthCard>
              <div className="space-y-5">
                <Field label="Email"><Input className="input-like" value={email} onChange={e=>setEmail(e.target.value)} placeholder="Enter email" /></Field>
                <Separator className="bg-white/5" />
                <AccentButton onClick={onSendResetCode} loading={loading}>Send confirmation code</AccentButton>
              </div>
            </AuthCard>
          </motion.div>
        )}

        {screen === "otp" && (
          <motion.div key="otp" variants={cardVariants} initial="initial" animate="animate" exit="exit">
            <AuthCard>
              <div className="space-y-5">
                <div>
                  <Label className="text-white/90">
                    {otpPurpose === "register" ? "Enter the verification code (Register)" : "Enter the confirmation code (Forgot password)"}
                  </Label>
                  <Input
                    className="input-like mt-2"
                    value={code}
                    onChange={(e) =>
                      setCode(
                        e.target.value.replace(/\D/g, "").slice(0, 6) // chỉ giữ số, tối đa 6 ký tự
                      )
                    }
                    placeholder="Enter the 6-digit code"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={6}
                    autoComplete="one-time-code"
                  />
                  <div className="mt-2 text-sm text-white/60">
                    <button className="hover:underline" onClick={resendOtp}>Resend code</button>
                  </div>
                </div>
                <Separator className="bg-white/5" />
                <AccentButton
                  onClick={otpPurpose === "register" ? verifyRegisterOtp : verifyResetOtp}
                  loading={loading}
                >
                  OK
                </AccentButton>
              </div>
            </AuthCard>
          </motion.div>
        )}

        {screen === "reset" && (
          <motion.div key="reset" variants={cardVariants} initial="initial" animate="animate" exit="exit">
            <AuthCard>
              <div className="space-y-5">
                <Field label="Reset Password"><Input className="input-like" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Enter password" type="password" /></Field>
                <Field label="Confirm password"><Input className="input-like" value={confirm} onChange={e=>setConfirm(e.target.value)} placeholder="Re-enter password" type="password" /></Field>
                <Separator className="bg-white/5" />
                <AccentButton onClick={onReset} loading={loading}>OK</AccentButton>
              </div>
            </AuthCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function AuthCard({ children }: { children: React.ReactNode }) {
  return (
    <Card className="bg-zinc-900/90 border border-white/5 rounded-2xl shadow-xl px-6 pt-8 pb-6">
      <CardHeader className="pb-6">
        <h1 className="mx-auto text-4xl font-extrabold tracking-tight">
          <span className="text-[#c8ff3d]">Fit</span><span>AI</span>
        </h1>
      </CardHeader>
      <CardContent>{children}</CardContent>
      <CardFooter />
    </Card>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <Label className="text-white/90">{label}</Label>
      <div className="mt-2">{children}</div>
    </div>
  )
}

function AccentButton({ children, onClick, loading }: { children: React.ReactNode; onClick: ()=>void; loading: boolean }) {
  return (
    <Button
      className="w-full h-11 rounded-full bg-[#c8ff3d] text-black font-semibold hover:opacity-90 shadow-[0_0_24px_rgba(200,255,61,0.35)]"
      onClick={onClick}
      disabled={loading}
    >
      {loading ? "Please wait..." : children}
    </Button>
  )
}
