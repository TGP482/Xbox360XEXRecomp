// the headset XamVoice gives the signed in player: the PC's microphone and speakers
#pragma once
#include <string>
#include <vector>

namespace xam {

// any thread: the PC's recording devices by name, as app::Settings::microphone keeps them
std::vector<std::string> Microphones();

}  // namespace xam
