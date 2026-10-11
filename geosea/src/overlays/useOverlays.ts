import { useState, useMemo } from 'react';
import { OVERLAYS } from './overlayConfig';

export interface OverlayState {
  enabled: boolean;
  opacity: number;
  date?: string;
  url?: string;
  bounds?: [[number, number], [number, number]];
}

export function useOverlays() {
  const isEnabledGlobal = import.meta.env.VITE_ENABLE_OVERLAYS !== 'false';

  const [masterSwitch, setMasterSwitch] = useState<boolean>(false);
  const [overlayStates, setOverlayStates] = useState<Record<string, OverlayState>>(() => {
    const initialState: Record<string, OverlayState> = {};
    for (const overlay of OVERLAYS) {
      initialState[overlay.id] = {
        enabled: overlay.defaultOn,
        opacity: overlay.defaultOpacity,
        date: overlay.defaultDate,
      };
    }
    return initialState;
  });

  const toggleOverlay = (id: string) => {
    setOverlayStates((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        enabled: !prev[id].enabled,
      }
    }));
  };

  const setOpacity = (id: string, opacity: number) => {
    setOverlayStates((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        opacity,
      }
    }));
  };

  const setDate = (id: string, date: string) => {
    setOverlayStates((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        date,
      }
    }));
  };

  const setUrl = (id: string, url: string) => {
    setOverlayStates((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        url,
      }
    }));
  };

  const setBounds = (id: string, bounds: [[number, number], [number, number]]) => {
    setOverlayStates((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        bounds,
      }
    }));
  };

  const toggleMasterSwitch = () => {
    setMasterSwitch((prev) => {
      const next = !prev;
      if (!next) {
        // Turn all off when master switch is off
        setOverlayStates((currentStates) => {
          const newStates = { ...currentStates };
          for (const key in newStates) {
            newStates[key].enabled = false;
          }
          return newStates;
        });
      }
      return next;
    });
  };

  const activeOverlays = useMemo(() => {
    if (!isEnabledGlobal || !masterSwitch) return [];
    return OVERLAYS.filter((overlay) => overlayStates[overlay.id]?.enabled && overlay.isConfigured !== false).map(overlay => {
      const state = overlayStates[overlay.id];
      let url = state.url || overlay.url;
      if (url && state.date) {
        url = url.replace('{time}', state.date);
      }
      return { 
        ...overlay, 
        url,
        bounds: state.bounds || overlay.bounds 
      };
    });
  }, [isEnabledGlobal, masterSwitch, overlayStates]);

  return {
    isEnabledGlobal,
    masterSwitch,
    toggleMasterSwitch,
    overlayStates,
    toggleOverlay,
    setOpacity,
    setDate,
    setUrl,
    setBounds,
    activeOverlays,
  };
}
