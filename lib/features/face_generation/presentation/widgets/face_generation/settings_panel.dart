import 'dart:ui';
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';
import 'package:psych_gen_app/core/designsystem/widgets/custom_number_text_field.dart';

class SettingsPanel extends StatelessWidget {
  final bool preserveIdentity;
  final double truncationPsi;
  final String mode;
  final ValueChanged<bool> onPreserveIdentityChanged;
  final ValueChanged<double> onTruncationPsiChanged;
  final ValueChanged<String> onModeChanged;
  final ValueChanged<int> onNumFacesChanged;
  final VoidCallback onGenerateDatasetPressed;
  final bool showTruncation;

  const SettingsPanel({
    super.key,
    required this.preserveIdentity,
    required this.truncationPsi,
    required this.mode,
    required this.onPreserveIdentityChanged,
    required this.onTruncationPsiChanged,
    required this.onModeChanged,
    required this.onNumFacesChanged,
    required this.onGenerateDatasetPressed,
    this.showTruncation = true,
  });

  Widget _buildModeSegment(
      BuildContext context, String value, String other, bool isDark) {
    final scheme = Theme.of(context).colorScheme;
    final selected = mode == 'both' || mode == value;
    final brandColor = isDark ? Colors.white : const Color(0xFF2B3A55);
    final onBrandColor = isDark ? const Color(0xFF2B3A55) : Colors.white;

    return Expanded(
      child: Semantics(
        toggled: selected,
        child: GestureDetector(
          onTap: () => onModeChanged(selected ? other : 'both'),
          behavior: HitTestBehavior.opaque,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeInOut,
            height: 36,
            decoration: BoxDecoration(
              color: selected ? brandColor : Colors.transparent,
              borderRadius: BorderRadius.circular(7),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  selected ? Icons.check_circle_rounded : Icons.circle_outlined,
                  size: 15,
                  color: selected ? onBrandColor : scheme.onSurfaceVariant,
                ),
                const SizedBox(width: 6),
                Text(
                  'settings.$value'.tr(),
                  style: TextStyle(
                    fontFamily: 'WorkSans',
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: selected ? onBrandColor : scheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final brandColor = isDark ? Colors.white : const Color(0xFF2B3A55);
    final onBrandColor = isDark ? const Color(0xFF2B3A55) : Colors.white;

    return ExpansionTile(
      initiallyExpanded: true,
      maintainState: true,
      title: Text(
        'section.settings'.tr(),
        style: const TextStyle(
          fontFamily: 'WorkSans',
          fontSize: 13,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.5,
        ),
      ),
      children: <Widget>[
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Preserve Identity — subtle 1px border row
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: scheme.outlineVariant, width: 1),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'settings.preserve_identity'.tr(),
                      style: const TextStyle(
                        fontFamily: 'WorkSans',
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    Tooltip(
                      message: 'tooltip.preserve_identity'.tr(),
                      child: Switch(
                        value: preserveIdentity,
                        onChanged: onPreserveIdentityChanged,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // Mode Toggle
              Text(
                'settings.mode'.tr(),
                style: const TextStyle(
                  fontFamily: 'WorkSans',
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(3),
                decoration: BoxDecoration(
                  border: Border.all(color: scheme.outlineVariant),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  children: [
                    _buildModeSegment(context, 'shape', 'color', isDark),
                    const SizedBox(width: 2),
                    _buildModeSegment(context, 'color', 'shape', isDark),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // Truncation Psi
              if (showTruncation) ...[
                Text(
                  'settings.truncation_psi'.tr(),
                  style: const TextStyle(
                    fontFamily: 'WorkSans',
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Row(
                  children: [
                    Expanded(
                      child: Tooltip(
                        message: 'tooltip.truncation_psi'.tr(),
                        child: Slider(
                          value: truncationPsi,
                          min: 0.1,
                          max: 1.0,
                          divisions: 9,
                          onChanged: onTruncationPsiChanged,
                        ),
                      ),
                    ),
                    SizedBox(
                      width: 50,
                      child: Text(
                        truncationPsi.toStringAsFixed(1),
                        style: const TextStyle(
                          fontFamily: 'WorkSans',
                          fontSize: 16,
                          fontFeatures: [FontFeature.tabularFigures()],
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
              ],

              // Number of images — subtle 1px border
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: scheme.outlineVariant, width: 1),
                ),
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'settings.num_images_each'.tr(),
                      style: const TextStyle(
                        fontFamily: 'WorkSans',
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 10),
                    CustomNumberTextField(onChanged: (numberOfFaces) {
                      if (numberOfFaces != null) onNumFacesChanged(numberOfFaces);
                    }),
                    const SizedBox(height: 8),
                    Text(
                      'settings.total_images_info'.tr(),
                      style: TextStyle(
                        fontFamily: 'WorkSans',
                        fontSize: 12,
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // Generate dataset button
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  elevation: 0,
                  backgroundColor: brandColor,
                  foregroundColor: onBrandColor,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  shape: const StadiumBorder(),
                ),
                icon: const Icon(Icons.download_rounded, size: 18),
                onPressed: onGenerateDatasetPressed,
                label: Text(
                  'button.generate_dataset'.tr(),
                  style: const TextStyle(
                    fontFamily: 'WorkSans',
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        )
      ],
    );
  }
}
