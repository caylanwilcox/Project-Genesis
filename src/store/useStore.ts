import { create } from 'zustand';
import { Signal, Engine, MarketData, Report } from '../types/Signal';

interface AppState {
  signals: Signal[];
  engines: Engine[];
  marketData: Map<string, MarketData[]>;
  reports: Report[];
  selectedSymbol: string | null;
  selectedEngine: string | null;
  regime: 'bull' | 'bear' | 'neutral';

  setSignals: (signals: Signal[]) => void;
  addSignal: (signal: Signal) => void;
  setEngines: (engines: Engine[]) => void;
  updateEngine: (engineId: string, updates: Partial<Engine>) => void;
  setMarketData: (symbol: string, data: MarketData[]) => void;
  setReports: (reports: Report[]) => void;
  addReport: (report: Report) => void;
  setSelectedSymbol: (symbol: string | null) => void;
  setSelectedEngine: (engine: string | null) => void;
  setRegime: (regime: 'bull' | 'bear' | 'neutral') => void;
}

export const useStore = create<AppState>((set) => ({
  signals: [],
  engines: [],
  marketData: new Map(),
  reports: [],
  selectedSymbol: null,
  selectedEngine: null,
  regime: 'neutral',

  setSignals: (signals) => set({ signals }),
  addSignal: (signal) => set((state) => ({ signals: [...state.signals, signal] })),
  setEngines: (engines) => set({ engines }),
  updateEngine: (engineId, updates) => set((state) => ({
    engines: state.engines.map(e => e.id === engineId ? { ...e, ...updates } : e)
  })),
  setMarketData: (symbol, data) => set((state) => {
    const newData = new Map(state.marketData);
    newData.set(symbol, data);
    return { marketData: newData };
  }),
  setReports: (reports) => set({ reports }),
  addReport: (report) => set((state) => ({ reports: [...state.reports, report] })),
  setSelectedSymbol: (selectedSymbol) => set({ selectedSymbol }),
  setSelectedEngine: (selectedEngine) => set({ selectedEngine }),
  setRegime: (regime) => set({ regime }),
}));