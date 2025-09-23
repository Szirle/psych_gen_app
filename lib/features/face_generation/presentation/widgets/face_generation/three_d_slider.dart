import 'package:flutter/material.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';

class ThreeDLevelSlider extends StatelessWidget {
  final ManipulatedDimension? sliderDim;
  final int sliderValue;
  final ValueChanged<int> onChanged;

  const ThreeDLevelSlider({
    super.key,
    required this.sliderDim,
    required this.sliderValue,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    if (sliderDim == null) {
      return const SizedBox.shrink();
    }
    return Container(
      padding: const EdgeInsets.all(16.0),
      margin: const EdgeInsets.only(bottom: 16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey[300]!, width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "${sliderDim!.name.name} Level",
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  fontFamily: 'WorkSans',
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF2B3A55),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  "Level $sliderValue",
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: const Color(0xFF2B3A55),
              inactiveTrackColor: Colors.grey[300],
              thumbColor: const Color(0xFF2B3A55),
              overlayColor: const Color(0xFF2B3A55).withOpacity(0.2),
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
              trackHeight: 4.0,
              valueIndicatorColor: const Color(0xFF2B3A55),
              valueIndicatorTextStyle: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
            child: Slider(
              value: sliderValue.toDouble(),
              min: 1,
              max: sliderDim!.nLevels.toDouble(),
              divisions: sliderDim!.nLevels > 1 ? sliderDim!.nLevels - 1 : 1,
              label: "Level $sliderValue",
              onChanged: (newValue) => onChanged(newValue.round()),
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "Level 1",
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.grey[600],
                ),
              ),
              Text(
                "Level ${sliderDim!.nLevels}",
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
