import { useCallback, useState } from 'react'

export function usePythonRPC() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const call = useCallback(async (method: string, params?: any) => {
    setLoading(true)
    setError(null)
    try {
      const result = await window.api.pythonCall(method, params)
      return result
    } catch (err: any) {
      setError(err.message || String(err))
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  return { call, loading, error }
}
