# GRPO Conceptual Consistency

Bu depo, çoktan seçmeli soru cevaplama görevlerinde şanslı tahminleri kavramsal olarak tutarlı doğru cevaplardan ayırmayı amaçlayan GRPO tabanlı reward tasarımına ait kodları, verileri ve deney sonuçlarını içermektedir.

## Amaç

Standart GRPO yaklaşımında doğru cevaplar eşit ödüllendirilir. Bu çalışmada, modelin özgün soruyla aynı kavramı ölçen benzer bir sorudaki performansı da dikkate alınarak doğru cevaplar farklı reward seviyeleriyle değerlendirilmiştir.

## Deney Koşulları

- **mA:** Standart binary reward
- **mB:** Kavramsal tutarlılık tabanlı reward
- **Modeller:** Qwen2.5-3B-Instruct ve Qwen3-4B
- **Veri Seti:** MMLU-TR v0.2
- **Görev:** Türkçe çoktan seçmeli soru cevaplama
