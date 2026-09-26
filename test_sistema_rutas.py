"""Pruebas automaticas del sistema. Ejecutar con: python -m unittest test_sistema_rutas -v"""
import unittest
from sistema_rutas import Planificador


class PruebasSistemaRutas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Planificador()

    def test_ruta_sin_transbordo(self):
        r = self.p.buscar_ruta("estadio_plazas_alcid", "parque_santander")
        self.assertEqual(len(r["transbordos"]), 0)

    def test_ruta_con_un_transbordo(self):
        r = self.p.buscar_ruta("terminal_transporte", "aeropuerto_benito_salas")
        self.assertEqual(len(r["transbordos"]), 1)

    def test_ruta_con_dos_transbordos(self):
        r = self.p.buscar_ruta("aeropuerto_benito_salas", "universidad_surcolombiana")
        self.assertEqual(len(r["transbordos"]), 2)
        self.assertEqual(r["minutos"], 39.0)

    def test_simetria(self):
        a = self.p.buscar_ruta("terminal_transporte", "universidad_surcolombiana")["minutos"]
        b = self.p.buscar_ruta("universidad_surcolombiana", "terminal_transporte")["minutos"]
        self.assertAlmostEqual(a, b, places=1)

    def test_astar_igual_resultado_que_sin_heuristica(self):
        casos = [("aeropuerto_benito_salas", "universidad_surcolombiana"),
                 ("terminal_transporte", "aeropuerto_benito_salas"),
                 ("san_pedro_plaza", "universidad_surcolombiana")]
        for o, d in casos:
            con_h = self.p.buscar_ruta(o, d)
            sin_h = self.p.buscar_ruta(o, d, usar_heuristica=False)
            self.assertAlmostEqual(con_h["minutos"], sin_h["minutos"], places=1)

    def test_astar_explora_menos_o_igual_nodos(self):
        con_h = self.p.buscar_ruta("aeropuerto_benito_salas", "universidad_surcolombiana")
        sin_h = self.p.buscar_ruta("aeropuerto_benito_salas", "universidad_surcolombiana", usar_heuristica=False)
        self.assertLess(con_h["nodos_expandidos"], sin_h["nodos_expandidos"])

    def test_reglas_deducen_transbordo_correcto(self):
        self.assertTrue(self.p.kb.existe("transbordo", "parque_santander", "ruta_1", "ruta_2")
                         or self.p.kb.existe("transbordo", "parque_santander", "ruta_2", "ruta_1"))
        self.assertFalse(self.p.kb.existe("transbordo", "san_pedro_plaza", "ruta_1", "ruta_2"))

    def test_parada_inexistente_lanza_error(self):
        with self.assertRaises(ValueError):
            self.p.buscar_ruta("lugar_que_no_existe", "parque_santander")


if __name__ == "__main__":
    unittest.main(verbosity=2)
