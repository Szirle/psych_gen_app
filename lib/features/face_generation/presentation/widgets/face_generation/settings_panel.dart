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
  });

  @override
  Widget build(BuildContext context) {
    return ExpansionTile(
      initiallyExpanded: true,
      maintainState: true,
      title: Text(
        'section.settings'.tr(),
        style: const TextStyle(
          fontFamily: 'WorkSans',
          fontWeight: FontWeight.bold,
        ),
      ),
      children: <Widget>[
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 12),
          child: Column(children: [
            Text('settings.preserve_identity'.tr(),
                style: const TextStyle(fontSize: 12)),
            Center(
                child: Tooltip(
              message: 'tooltip.preserve_identity'.tr(),
              child: Switch(
                value: preserveIdentity,
                onChanged: onPreserveIdentityChanged,
                activeColor: const Color(0xFF2B3A55),
              ),
            )),
            const SizedBox(height: 10),
            Text('settings.truncation_psi'.tr(),
                style: const TextStyle(fontSize: 12)),
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
                    style: const TextStyle(fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            const SizedBox(height: 10),
            Text('settings.mode'.tr(), style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 10),
            SizedBox(
                height: 36,
                child: DropdownButtonFormField<String>(
                  decoration: InputDecoration(
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide:
                          const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide:
                          const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    contentPadding:
                        const EdgeInsets.only(top: 12, left: 12, right: 12),
                  ),
                  value: mode,
                  onChanged: (String? newValue) {
                    if (newValue != null) onModeChanged(newValue);
                  },
                  items: ['shape', 'color', 'both']
                      .map<DropdownMenuItem<String>>((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(
                        value,
                        style: const TextStyle(
                          fontSize: 14,
                          fontFamily: 'WorkSans',
                        ),
                      ),
                    );
                  }).toList(),
                )),
            const SizedBox(height: 20),
            Text('settings.num_images_each'.tr(),
                style: const TextStyle(fontSize: 12)),
            const SizedBox(width: 20),
            CustomNumberTextField(onChanged: (numberOfFaces) {
              if (numberOfFaces != null) onNumFacesChanged(numberOfFaces);
            }),
            Text('settings.total_images_info'.tr(),
                style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 20),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                elevation: 0,
                backgroundColor: const Color(0xFF2B3A55),
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(5),
                ),
              ),
              onPressed: onGenerateDatasetPressed,
              child: Text(
                'button.generate_dataset'.tr(),
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ]),
        )
      ],
    );
  }
}
