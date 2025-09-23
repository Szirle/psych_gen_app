import 'dart:math' as math;
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';

class ChartsAnchor extends StatelessWidget {
  final bool isOpen;
  final VoidCallback onTap;

  const ChartsAnchor({super.key, required this.isOpen, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: isOpen ? 'panel.hide_charts'.tr() : 'panel.show_charts'.tr(),
      child: GestureDetector(
        onTap: onTap,
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          child: Container(
            width: 22,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(8),
                bottomLeft: Radius.circular(8),
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.08),
                  blurRadius: 6,
                  offset: const Offset(0, 2),
                ),
              ],
              border: Border.all(color: Colors.grey[300]!, width: 1),
            ),
            child: Transform.rotate(
              angle: isOpen ? math.pi : 0,
              child: const Icon(
                Icons.chevron_left,
                size: 18,
                color: Color(0xFF2B3A55),
              ),
            ),
          ),
        ),
      ),
    );
  }
}


