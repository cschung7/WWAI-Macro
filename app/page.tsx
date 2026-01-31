'use client'

import { useState } from 'react'
import Link from 'next/link'
import { translations, countries, Lang } from './translations'

// Local VAR Dashboard (GraphEconCast with VAR model)
// Features: Investment Dashboard, Full VAR (Diebold-Yilmaz Spillover, BMA, Regime Switching)
const VAR_DASHBOARD = 'http://163.239.155.96:8012'
// Local GNN Dashboard (GraphEconCast with GNN model)
// Features: GNN-based spillover analysis, message passing visualization
const GNN_DASHBOARD = 'http://163.239.155.96:3789'
// Cache-busting version - increment when dashboard JS changes
const CACHE_VERSION = 'v4'

export default function LandingPage() {
  const [lang, setLang] = useState<Lang>('en')
  const t = translations[lang]

  // Quick scenario state
  const [shockCountry, setShockCountry] = useState('USA')
  const [shockVariable, setShockVariable] = useState('interest_rate')
  const [shockMagnitude, setShockMagnitude] = useState(50)
  const [loading, setLoading] = useState(false)

  const getCountryName = (code: string) => {
    const country = countries.find(c => c.code === code)
    return lang === 'ko' ? country?.name_ko : country?.name_en
  }

  const handleRunBoth = () => {
    // Open main dashboard with shock simulator parameters
    const dashboardUrl = `${VAR_DASHBOARD}/?${CACHE_VERSION}&country=${shockCountry}&variable=${shockVariable}&magnitude=${shockMagnitude}`
    window.open(dashboardUrl, '_blank')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 backdrop-blur-sm bg-slate-900/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
              <span className="text-white font-bold text-lg">W</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">{t.title}</h1>
              <p className="text-xs text-slate-400">{t.subtitle}</p>
            </div>
          </div>

          <nav className="flex items-center gap-6">
            <Link href="/" className="text-slate-300 hover:text-white transition-colors text-sm">{t.nav.home}</Link>
            <Link href="/compare" className="text-slate-300 hover:text-white transition-colors text-sm">{t.nav.compare}</Link>
            <a href={`${GNN_DASHBOARD}/spillovers`} target="_blank" className="text-slate-300 hover:text-white transition-colors text-sm">{t.nav.gnn}</a>
            <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
              <button
                onClick={() => setLang('en')}
                className={`px-3 py-1 rounded text-sm transition-all ${lang === 'en' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                EN
              </button>
              <button
                onClick={() => setLang('ko')}
                className={`px-3 py-1 rounded text-sm transition-all ${lang === 'ko' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                KO
              </button>
            </div>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-4 py-2 mb-6">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
            <span className="text-blue-400 text-sm">Live Economic Analysis</span>
          </div>

          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent">
              {t.title}
            </span>
          </h1>

          <p className="text-xl text-slate-300 mb-4">{t.subtitle}</p>
          <p className="text-slate-400 max-w-2xl mx-auto">{t.description}</p>
        </div>
      </section>

      {/* Model Cards */}
      <section className="py-12 px-4">
        <div className="max-w-7xl mx-auto grid md:grid-cols-3 gap-8">
          {/* GraphEconCast Investment Dashboard Card */}
          <div className="group relative">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative bg-slate-800/50 border border-slate-700 rounded-2xl p-8 hover:border-purple-500/50 transition-all">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <span className="text-2xl">📊</span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-2xl font-bold text-white">
                      {lang === 'en' ? 'Investment Dashboard' : '투자 대시보드'}
                    </h3>
                    <span className="px-2 py-0.5 text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full">
                      Local
                    </span>
                  </div>
                  <p className="text-purple-400">
                    {lang === 'en' ? 'Risk Analytics + Commodity Regime' : '리스크 분석 + 원자재 레짐'}
                  </p>
                </div>
              </div>

              <p className="text-slate-300 mb-6">
                {lang === 'en'
                  ? 'ML-powered commodity regime classification, VKOSPI analysis, and cross-asset equity signals. Requires NAS file access.'
                  : 'ML 기반 원자재 레짐 분류, VKOSPI 분석, 크로스 에셋 주식 신호. NAS 파일 접근 필요.'}
              </p>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-purple-400 font-mono text-sm">
                    {lang === 'en' ? '4 Regimes' : '4 레짐'}
                  </div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-purple-400 font-mono text-sm">VKOSPI</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-purple-400 font-mono text-sm">
                    {lang === 'en' ? 'Real-time' : '실시간'}
                  </div>
                </div>
              </div>

              <ul className="space-y-2 mb-6">
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-purple-400">+</span>
                  {lang === 'en' ? 'Commodity Regime ML (Bull/Bear Quiet/Volatile)' : '원자재 레짐 ML (강세/약세 안정/변동)'}
                </li>
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-purple-400">+</span>
                  {lang === 'en' ? "Korea's Fear Gauge (VKOSPI)" : '한국 공포지수 (VKOSPI)'}
                </li>
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-purple-400">+</span>
                  {lang === 'en' ? 'Cross-Asset Equity Signals' : '크로스 에셋 주식 신호'}
                </li>
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-purple-400">+</span>
                  {lang === 'en' ? 'Position Sizing Recommendations' : '포지션 사이징 추천'}
                </li>
              </ul>

              <a
                href={`${VAR_DASHBOARD}/dashboard?${CACHE_VERSION}`}
                target="_blank"
                className="block w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold rounded-xl text-center hover:from-purple-600 hover:to-pink-600 transition-all"
              >
                {lang === 'en' ? 'Open Dashboard' : '대시보드 열기'} &rarr;
              </a>
            </div>
          </div>

          {/* VAR Card - Full Model (Local GraphEconCast) */}
          <div className="group relative">
            <div className="absolute inset-0 bg-gradient-to-r from-amber-500/20 to-amber-600/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative bg-slate-800/50 border border-slate-700 rounded-2xl p-8 hover:border-amber-500/50 transition-all">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center">
                  <span className="text-2xl">📈</span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-2xl font-bold text-white">{t.varModel.title}</h3>
                    <span className="px-2 py-0.5 text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full">
                      Local
                    </span>
                  </div>
                  <p className="text-amber-400">{t.varModel.subtitle}</p>
                </div>
              </div>

              <p className="text-slate-300 mb-6">
                {lang === 'en'
                  ? 'Full VAR model with Diebold-Yilmaz Spillover Index, Bayesian Model Averaging, and Regime Switching. Requires NAS file access.'
                  : '전체 VAR 모델: Diebold-Yilmaz 파급효과 지수, 베이지안 모델 평균, 레짐 스위칭. NAS 파일 접근 필요.'}
              </p>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-amber-400 font-mono text-sm">
                    {lang === 'en' ? 'Spillover' : '파급효과'}
                  </div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-amber-400 font-mono text-sm">BMA</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-amber-400 font-mono text-sm">
                    {lang === 'en' ? 'Regime' : '레짐'}
                  </div>
                </div>
              </div>

              <ul className="space-y-2 mb-6">
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-amber-400">+</span>
                  {lang === 'en' ? 'Diebold-Yilmaz Spillover Index' : 'Diebold-Yilmaz 파급효과 지수'}
                </li>
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-amber-400">+</span>
                  {lang === 'en' ? 'Bayesian Model Averaging (6 models)' : '베이지안 모델 평균 (6개 모델)'}
                </li>
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-amber-400">+</span>
                  {lang === 'en' ? 'Markov Regime Switching' : '마코프 레짐 스위칭'}
                </li>
                <li className="flex items-center gap-2 text-slate-300">
                  <span className="text-amber-400">+</span>
                  {lang === 'en' ? 'Structural Shock Identification' : '구조적 충격 식별'}
                </li>
              </ul>

              <a
                href={`${VAR_DASHBOARD}/?${CACHE_VERSION}#shock-simulator`}
                target="_blank"
                className="block w-full py-3 bg-gradient-to-r from-amber-500 to-amber-600 text-white font-semibold rounded-xl text-center hover:from-amber-600 hover:to-amber-700 transition-all"
              >
                {t.varModel.cta} &rarr;
              </a>
            </div>
          </div>

          {/* GNN Card - Local GraphEconCast */}
          <div className="group relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative bg-slate-800/50 border border-slate-700 rounded-2xl p-8 hover:border-blue-500/50 transition-all">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                  <span className="text-2xl">🕸️</span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-2xl font-bold text-white">{t.gnnModel.title}</h3>
                    <span className="px-2 py-0.5 text-xs font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 rounded-full">
                      Local
                    </span>
                  </div>
                  <p className="text-cyan-400">{t.gnnModel.subtitle}</p>
                </div>
              </div>

              <p className="text-slate-300 mb-6">
                {lang === 'en'
                  ? 'Graph Neural Network for economic spillover analysis. Multi-hop message passing captures complex cross-country dynamics. Requires NAS file access.'
                  : '경제 파급효과 분석을 위한 그래프 신경망. 멀티홉 메시지 패싱으로 복잡한 국가간 역학 포착. NAS 파일 접근 필요.'}
              </p>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-cyan-400 font-mono text-sm">{t.gnnModel.stats.r2}</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-cyan-400 font-mono text-sm">{t.gnnModel.stats.params}</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                  <div className="text-cyan-400 font-mono text-sm">{t.gnnModel.stats.edges}</div>
                </div>
              </div>

              <ul className="space-y-2 mb-6">
                {t.gnnModel.features.map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-slate-300">
                    <span className="text-cyan-400">+</span>
                    {feature}
                  </li>
                ))}
              </ul>

              <a
                href={`${GNN_DASHBOARD}/spillovers`}
                target="_blank"
                className="block w-full py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-semibold rounded-xl text-center hover:from-blue-600 hover:to-cyan-600 transition-all"
              >
                {t.gnnModel.cta} &rarr;
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Scenario Section */}
      <section className="py-12 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="bg-slate-800/30 border border-slate-700 rounded-2xl p-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-white mb-2">{t.quickScenario.title}</h2>
              <p className="text-slate-400">{t.quickScenario.description}</p>
            </div>

            <div className="grid md:grid-cols-4 gap-4 mb-6">
              <div>
                <label className="block text-sm text-slate-400 mb-2">{t.quickScenario.shockCountry}</label>
                <select
                  value={shockCountry}
                  onChange={(e) => setShockCountry(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:border-blue-500 focus:outline-none"
                >
                  {countries.map(c => (
                    <option key={c.code} value={c.code}>
                      {lang === 'ko' ? c.name_ko : c.name_en}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-2">{t.quickScenario.shockVariable}</label>
                <select
                  value={shockVariable}
                  onChange={(e) => setShockVariable(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:border-blue-500 focus:outline-none"
                >
                  {Object.entries(t.quickScenario.variables).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-2">{t.quickScenario.shockMagnitude} (bp)</label>
                <input
                  type="number"
                  value={shockMagnitude}
                  onChange={(e) => setShockMagnitude(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:border-blue-500 focus:outline-none"
                  step={25}
                />
              </div>

              <div className="flex items-end">
                <button
                  onClick={handleRunBoth}
                  disabled={loading}
                  className="w-full py-3 bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold rounded-lg hover:from-emerald-600 hover:to-teal-600 transition-all disabled:opacity-50"
                >
                  {loading ? '...' : t.quickScenario.runBoth}
                </button>
              </div>
            </div>

            <div className="text-center text-slate-500 text-sm">
              {lang === 'en'
                ? `Scenario: ${getCountryName(shockCountry)} ${shockVariable.replace(/_/g, ' ')} shock of ${shockMagnitude >= 0 ? '+' : ''}${shockMagnitude}bp`
                : `시나리오: ${getCountryName(shockCountry)} ${t.quickScenario.variables[shockVariable as keyof typeof t.quickScenario.variables]} ${shockMagnitude >= 0 ? '+' : ''}${shockMagnitude}bp 충격`
              }
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-12 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-2xl font-bold text-white text-center mb-8">{t.features.title}</h2>

          <div className="grid md:grid-cols-4 gap-6">
            {t.features.items.map((item, i) => (
              <div key={i} className="bg-slate-800/30 border border-slate-700 rounded-xl p-6 text-center hover:border-slate-600 transition-all">
                <div className="text-4xl mb-4">{item.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-slate-400 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Preview */}
      <section className="py-12 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="bg-gradient-to-r from-slate-800/50 to-slate-700/50 border border-slate-600 rounded-2xl p-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-2">{t.comparison.title}</h2>
                <p className="text-slate-400">{t.comparison.subtitle}</p>
              </div>
              <Link
                href="/compare"
                className="px-8 py-3 bg-white text-slate-900 font-semibold rounded-xl hover:bg-slate-100 transition-all"
              >
                {lang === 'en' ? 'Open Comparison Tool' : '비교 도구 열기'} &rarr;
              </Link>
            </div>

            <div className="grid md:grid-cols-2 gap-6 mt-8">
              <div className="bg-slate-900/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-amber-400 mb-4">{t.comparison.varResults}</h3>
                <div className="space-y-3 text-slate-300 text-sm">
                  <div className="flex justify-between">
                    <span>{t.comparison.methodology}:</span>
                    <span className="text-slate-400">Least Squares → IRF</span>
                  </div>
                  <div className="flex justify-between">
                    <span>{t.comparison.strengths}:</span>
                    <span className="text-slate-400">{lang === 'en' ? 'Interpretable, Fast' : '해석 가능, 빠름'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>{t.comparison.limitations}:</span>
                    <span className="text-slate-400">{lang === 'en' ? 'Linear only' : '선형만'}</span>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-cyan-400 mb-4">{t.comparison.gnnResults}</h3>
                <div className="space-y-3 text-slate-300 text-sm">
                  <div className="flex justify-between">
                    <span>{t.comparison.methodology}:</span>
                    <span className="text-slate-400">Message Passing × 8</span>
                  </div>
                  <div className="flex justify-between">
                    <span>{t.comparison.strengths}:</span>
                    <span className="text-slate-400">{lang === 'en' ? 'Non-linear, Multi-hop' : '비선형, 다중홉'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>{t.comparison.limitations}:</span>
                    <span className="text-slate-400">{lang === 'en' ? 'Less interpretable' : '해석 어려움'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-8 px-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-slate-500 text-sm">{t.footer.copyright}</div>
          <div className="flex items-center gap-4 text-slate-500 text-sm">
            <span>{t.footer.version}</span>
            <span>|</span>
            <span>All Models: Local (NAS)</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
