export function readStorage<T>(key: string, fallback: T): T {
  const raw = window.localStorage.getItem(key);
  if (raw === null) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch (error) {
    throw new Error(`Could not read saved ${key} data. Clear the corrupted browser data and try again.`, { cause: error });
  }
}

export function writeStorage<T>(key: string, value: T): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    throw new Error("Could not save your work to browser storage. Check available storage space and try again.", { cause: error });
  }
}
