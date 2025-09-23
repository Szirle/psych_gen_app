import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';

class PreviewHeaderBar extends StatelessWidget {
  final VoidCallback onChangeFacePressed;

  const PreviewHeaderBar({super.key, required this.onChangeFacePressed});

  @override
  Widget build(BuildContext context) {
    return Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(12, 0, 0, 0),
        child: Text(
          'preview.title'.tr(),
          style: const TextStyle(
              fontFamily: 'WorkSans', fontSize: 28, color: Color(0xFF4A5568)),
        ),
      ),
      Padding(
        padding: const EdgeInsets.fromLTRB(12, 4, 0, 0),
        child: Text(
          'nav.breadcrumb'.tr(),
          style: const TextStyle(
              fontFamily: 'WorkSans', fontSize: 11, color: Color(0xFF4A5568)),
        ),
      ),
      Row(children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(0, 8, 8, 0),
          child: SizedBox(
            width: 140,
            child: Tooltip(
              message: 'tooltip.change_face'.tr(),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  elevation: 0,
                  backgroundColor: const Color(0xFF2B3A55),
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(5),
                  ),
                ),
                onPressed: onChangeFacePressed,
                child: Text('button.change_face'.tr(),
                    style: const TextStyle(color: Colors.white)),
              ),
            ),
          ),
        ),
      ])
    ]);
  }
}


