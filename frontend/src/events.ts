import { useEffect, useRef, useState } from "react";

import type { ExperimentSummary, RunSummary, Step } from "./api";

export interface CheckpointEvent {
  type: string;
  run?: RunSummary;
  step?: Step;
  experiment?: ExperimentSummary;
  experiment_id?: string;
  variant_name?: string;
  run_id?: string;
  passed?: boolean;
  completed?: number;
  total?: number;
  failed?: number;
}

/** Subscribe to daemon events and reconnect when the socket drops. */
export function useEvents(onEvent: (event: CheckpointEvent) => void): boolean {
  const handler = useRef(onEvent);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    handler.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let disposed = false;

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
      socket.onopen = () => setConnected(true);
      socket.onmessage = (message) => {
        try {
          handler.current(JSON.parse(message.data as string) as CheckpointEvent);
        } catch {
          // Ignore malformed events rather than breaking the page.
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!disposed) {
          retry = window.setTimeout(connect, 1500);
        }
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      disposed = true;
      if (retry !== undefined) {
        window.clearTimeout(retry);
      }
      socket?.close();
    };
  }, []);

  return connected;
}
