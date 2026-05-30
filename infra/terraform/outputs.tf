output "external_ip" {
  description = "Внешний IP VM со стеком"
  value       = yandex_compute_instance.uneemi.network_interface.0.nat_ip_address
}

output "service_urls" {
  description = "URL сервисов после применения (порты из .env)"
  value = {
    serving_health = "http://${yandex_compute_instance.uneemi.network_interface.0.nat_ip_address}:18000/health"
    airflow        = "http://${yandex_compute_instance.uneemi.network_interface.0.nat_ip_address}:18080"
    grafana        = "http://${yandex_compute_instance.uneemi.network_interface.0.nat_ip_address}:13000"
    mlflow         = "http://${yandex_compute_instance.uneemi.network_interface.0.nat_ip_address}:5500"
    prometheus     = "http://${yandex_compute_instance.uneemi.network_interface.0.nat_ip_address}:19090"
  }
}
