export type Lang = 'en' | 'ko'

export const translations = {
  en: {
    // Landing Page
    title: 'WWAI-Macro',
    subtitle: 'Global Macroeconomic Forecasting Platform',
    description: 'Advanced economic forecasting using VAR econometrics and Graph Neural Networks',

    // Model Cards
    varModel: {
      title: 'VAR Model',
      subtitle: 'Vector Autoregression',
      description: 'Traditional econometric time-series analysis with Impulse Response Functions and Markov Regime Switching',
      features: [
        'Linear Granger Causality',
        'Impulse Response Analysis',
        'Regime Detection (Bull/Bear)',
        'Diebold-Yilmaz Spillover Index'
      ],
      stats: {
        variables: '6-9 Variables',
        lag: '4 Quarter Lag',
        method: 'Least Squares'
      },
      cta: 'Launch VAR Dashboard'
    },
    gnnModel: {
      title: 'GNN Model',
      subtitle: 'Graph Neural Network',
      description: 'Deep learning approach with 8-step message passing for non-linear cross-country spillover effects',
      features: [
        'Non-linear Spillovers',
        'Learned Attention Weights',
        'Multi-hop Effects (8 steps)',
        '26 Country Network'
      ],
      stats: {
        r2: 'R² 99.49%',
        params: '4.03M Parameters',
        edges: '3 Edge Types'
      },
      cta: 'Launch GNN Dashboard'
    },

    // Quick Scenario
    quickScenario: {
      title: 'Quick Scenario Analysis',
      description: 'Run a shock simulation on both models simultaneously',
      shockCountry: 'Shock Origin',
      shockVariable: 'Variable',
      shockMagnitude: 'Magnitude',
      runBoth: 'Run Both Models',
      variables: {
        interest_rate: 'Interest Rate',
        gdp_growth_rate: 'GDP Growth',
        inflation_rate: 'Inflation',
        unemployment_rate: 'Unemployment',
        trade_balance: 'Trade Balance'
      }
    },

    // Comparison
    comparison: {
      title: 'Model Comparison',
      subtitle: 'VAR vs GNN for the same scenario',
      varResults: 'VAR Results (Linear IRF)',
      gnnResults: 'GNN Results (Non-linear MP)',
      methodology: 'Methodology',
      strengths: 'Strengths',
      limitations: 'Limitations'
    },

    // Features
    features: {
      title: 'Platform Features',
      items: [
        {
          icon: '🌍',
          title: 'Global Coverage',
          desc: '26 major economies including G20 nations'
        },
        {
          icon: '📊',
          title: 'Multi-Variable',
          desc: 'GDP, Inflation, Unemployment, Interest Rate, Trade'
        },
        {
          icon: '🔄',
          title: 'Real-time Analysis',
          desc: 'Live shock propagation simulation'
        },
        {
          icon: '📈',
          title: 'Bilingual Reports',
          desc: 'English and Korean report generation'
        }
      ]
    },

    // Navigation
    nav: {
      home: 'Home',
      var: 'VAR Analysis',
      gnn: 'GNN Spillovers',
      compare: 'Compare',
      docs: 'Documentation'
    },

    // Footer
    footer: {
      copyright: '© 2026 WWAI-Macro. All rights reserved.',
      version: 'Version 1.0.0'
    }
  },

  ko: {
    // Landing Page
    title: 'WWAI-Macro',
    subtitle: '글로벌 거시경제 예측 플랫폼',
    description: 'VAR 계량경제학과 그래프 신경망을 활용한 첨단 경제 예측',

    // Model Cards
    varModel: {
      title: 'VAR 모형',
      subtitle: '벡터자기회귀',
      description: '충격반응함수와 마코프 레짐 스위칭을 활용한 전통적 계량경제 시계열 분석',
      features: [
        '선형 그레인저 인과관계',
        '충격반응분석 (IRF)',
        '레짐 감지 (강세/약세)',
        'Diebold-Yilmaz 전이지수'
      ],
      stats: {
        variables: '6-9개 변수',
        lag: '4분기 시차',
        method: '최소자승법'
      },
      cta: 'VAR 대시보드 실행'
    },
    gnnModel: {
      title: 'GNN 모형',
      subtitle: '그래프 신경망',
      description: '8단계 메시지 패싱을 통한 비선형 국가 간 전이효과 분석 딥러닝 접근법',
      features: [
        '비선형 전이효과',
        '학습된 어텐션 가중치',
        '다중 홉 효과 (8단계)',
        '26개국 네트워크'
      ],
      stats: {
        r2: 'R² 99.49%',
        params: '403만 파라미터',
        edges: '3가지 엣지 유형'
      },
      cta: 'GNN 대시보드 실행'
    },

    // Quick Scenario
    quickScenario: {
      title: '빠른 시나리오 분석',
      description: '두 모형에서 동시에 충격 시뮬레이션 실행',
      shockCountry: '충격 발생국',
      shockVariable: '변수',
      shockMagnitude: '규모',
      runBoth: '두 모형 실행',
      variables: {
        interest_rate: '금리',
        gdp_growth_rate: 'GDP 성장률',
        inflation_rate: '인플레이션',
        unemployment_rate: '실업률',
        trade_balance: '무역수지'
      }
    },

    // Comparison
    comparison: {
      title: '모형 비교',
      subtitle: '동일 시나리오에 대한 VAR vs GNN',
      varResults: 'VAR 결과 (선형 IRF)',
      gnnResults: 'GNN 결과 (비선형 MP)',
      methodology: '방법론',
      strengths: '강점',
      limitations: '한계'
    },

    // Features
    features: {
      title: '플랫폼 기능',
      items: [
        {
          icon: '🌍',
          title: '글로벌 커버리지',
          desc: 'G20 포함 26개 주요 경제국'
        },
        {
          icon: '📊',
          title: '다변량 분석',
          desc: 'GDP, 인플레이션, 실업률, 금리, 무역'
        },
        {
          icon: '🔄',
          title: '실시간 분석',
          desc: '실시간 충격 전파 시뮬레이션'
        },
        {
          icon: '📈',
          title: '이중 언어 보고서',
          desc: '영어 및 한국어 보고서 생성'
        }
      ]
    },

    // Navigation
    nav: {
      home: '홈',
      var: 'VAR 분석',
      gnn: 'GNN 전이효과',
      compare: '비교',
      docs: '문서'
    },

    // Footer
    footer: {
      copyright: '© 2026 WWAI-Macro. All rights reserved.',
      version: '버전 1.0.0'
    }
  }
}

export const countries = [
  { code: 'USA', name_en: 'United States', name_ko: '미국' },
  { code: 'CHN', name_en: 'China', name_ko: '중국' },
  { code: 'JPN', name_en: 'Japan', name_ko: '일본' },
  { code: 'DEU', name_en: 'Germany', name_ko: '독일' },
  { code: 'GBR', name_en: 'United Kingdom', name_ko: '영국' },
  { code: 'FRA', name_en: 'France', name_ko: '프랑스' },
  { code: 'IND', name_en: 'India', name_ko: '인도' },
  { code: 'KOR', name_en: 'South Korea', name_ko: '한국' },
  { code: 'BRA', name_en: 'Brazil', name_ko: '브라질' },
  { code: 'CAN', name_en: 'Canada', name_ko: '캐나다' },
  { code: 'RUS', name_en: 'Russia', name_ko: '러시아' },
  { code: 'AUS', name_en: 'Australia', name_ko: '호주' },
  { code: 'MEX', name_en: 'Mexico', name_ko: '멕시코' },
  { code: 'IDN', name_en: 'Indonesia', name_ko: '인도네시아' },
  { code: 'SAU', name_en: 'Saudi Arabia', name_ko: '사우디아라비아' },
  { code: 'TUR', name_en: 'Turkey', name_ko: '터키' },
  { code: 'ARG', name_en: 'Argentina', name_ko: '아르헨티나' },
  { code: 'ZAF', name_en: 'South Africa', name_ko: '남아프리카' },
  { code: 'ITA', name_en: 'Italy', name_ko: '이탈리아' },
  { code: 'ESP', name_en: 'Spain', name_ko: '스페인' },
  { code: 'NLD', name_en: 'Netherlands', name_ko: '네덜란드' },
  { code: 'CHE', name_en: 'Switzerland', name_ko: '스위스' },
  { code: 'POL', name_en: 'Poland', name_ko: '폴란드' },
  { code: 'SWE', name_en: 'Sweden', name_ko: '스웨덴' },
  { code: 'BEL', name_en: 'Belgium', name_ko: '벨기에' },
  { code: 'THA', name_en: 'Thailand', name_ko: '태국' }
]
